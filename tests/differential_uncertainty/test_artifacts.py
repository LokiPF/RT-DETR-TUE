import builtins
import dis
import gc
import hashlib
import inspect
import json
import os
import pickle
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import torch

import differential_uncertainty.artifacts as artifacts
from differential_uncertainty.artifacts import (
    ShardWriter,
    atomic_json,
    atomic_torch,
    ensure_provenance,
    iter_records,
    load_manifest,
    sha256_file,
    source_digest,
)


def _record(image_id: str, severity: int = 0, value: int = 1) -> dict:
    return {
        "image_id": image_id,
        "severity": severity,
        "value": torch.tensor([value]),
    }


def _write_manual_artifact(root: Path, records: list[dict]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    name = "shard_00000.pt"
    shard = root / name
    torch.save(records, shard)
    atomic_json(
        {
            "schema_version": 1,
            "record_count": len(records),
            "shards": [name],
            "shard_sha256": {name: sha256_file(shard)},
        },
        root / "manifest.json",
    )
    return shard


def _traceable(function):
    return getattr(function, "__wrapped__", function)


def _interrupt_at_instruction(function, invoke, predicate, *, before_interrupt=None) -> None:
    function = _traceable(function)
    instructions = {
        instruction.offset: instruction
        for instruction in dis.get_instructions(function)
    }
    for _attempt in range(2):
        interrupted = False

        def interrupt(frame, event, _argument):
            nonlocal interrupted
            if frame.f_code is function.__code__:
                frame.f_trace_opcodes = True
                instruction = instructions.get(frame.f_lasti)
                if (
                    not interrupted
                    and event == "opcode"
                    and instruction is not None
                    and predicate(frame, instruction)
                ):
                    interrupted = True
                    if before_interrupt is not None:
                        before_interrupt(frame, instruction)
                    raise KeyboardInterrupt(
                        "interrupted at file ownership boundary"
                    )
            return interrupt

        try:
            sys.settrace(interrupt)
            invoke()
        except KeyboardInterrupt as error:
            if not interrupted:
                raise
            assert "file ownership boundary" in str(error)
        finally:
            sys.settrace(None)
        if interrupted:
            return
    pytest.fail("ownership boundary was not executed")


def _interrupt_at_opcode(
    function,
    opname,
    argval,
    invoke,
    *,
    occurrence=0,
) -> None:
    function = _traceable(function)
    targets = [
        instruction.offset
        for instruction in dis.get_instructions(function)
        if instruction.opname == opname
        and (argval is None or instruction.argval == argval)
    ]
    target = targets[occurrence]
    _interrupt_at_instruction(
        function,
        invoke,
        lambda _frame, instruction: instruction.offset == target,
    )


def _attribute_call_return_offsets(function, attribute: str) -> set[int]:
    instructions = list(dis.get_instructions(_traceable(function)))
    boundaries = set()
    for index, instruction in enumerate(instructions):
        if instruction.argval != attribute or instruction.opname not in (
            "LOAD_ATTR",
            "LOAD_METHOD",
        ):
            continue
        for call_index in range(index + 1, len(instructions) - 1):
            if instructions[call_index].opname.startswith("CALL"):
                boundaries.add(instructions[call_index + 1].offset)
                break
    return boundaries


def _temporary_entries(root: Path) -> list[Path]:
    return [path for path in root.iterdir() if path.name.endswith(".tmp")]


def _publication_case(tmp_path: Path, kind: str):
    root = tmp_path / kind
    root.mkdir()
    directory_fd = None
    if kind == "public_json":
        invoke = lambda: artifacts.atomic_json({"value": 1}, root / "value.json")
    elif kind == "public_json_create":
        invoke = lambda: artifacts._atomic_json_create(
            {"value": 1}, root / "value.json"
        )
    elif kind == "public_torch":
        invoke = lambda: artifacts.atomic_torch(
            {"value": torch.tensor([1])}, root / "value.pt"
        )
    else:
        directory_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        if kind == "at_json":
            invoke = lambda: artifacts._atomic_json_at(
                {"value": 1}, "value.json", directory_fd
            )
        else:
            invoke = lambda: artifacts._atomic_torch_at(
                {"value": torch.tensor([1])}, "value.pt", directory_fd
            )
    return root, directory_fd, invoke


def _read_case(tmp_path: Path, kind: str):
    root = tmp_path / kind
    root.mkdir()
    directory_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    if kind == "read_json":
        (root / "value.json").write_text('{"value": 1}\n', encoding="utf-8")
        invoke = lambda: artifacts._read_json_at(directory_fd, "value.json")
    elif kind == "sha256_at":
        (root / "value.bin").write_bytes(b"value")
        invoke = lambda: artifacts._sha256_file_at(directory_fd, "value.bin")
    else:
        shard = root / "shard_00000.pt"
        torch.save([_record("a")], shard)
        digest = sha256_file(shard)
        if kind == "shard_path":
            invoke = lambda: artifacts._safe_load_shard(
                root, shard.name, digest
            )
        else:
            invoke = lambda: artifacts._safe_load_shard(
                root,
                shard.name,
                digest,
                directory_fd=directory_fd,
            )
    return root, directory_fd, invoke


@pytest.mark.parametrize(
    "kind",
    ["public_json", "public_json_create", "public_torch", "at_json", "at_torch"],
)
def test_staging_open_return_interrupt_has_no_raw_descriptor_or_file(
    tmp_path,
    monkeypatch,
    kind,
):
    root, directory_fd, invoke = _publication_case(tmp_path, kind)
    original_close = os.close
    observed: list[tuple[int, Path]] = []

    if hasattr(artifacts, "_staged_file"):
        target = artifacts._staged_file
        opname, argval = "STORE_FAST", "handle"
        original_open = builtins.open

        def observed_open(path, mode="r", *args, **kwargs):
            handle = original_open(path, mode, *args, **kwargs)
            if mode == "xb" and os.fspath(path).endswith(".tmp"):
                observed.append((handle.fileno(), Path(path)))
            return handle

        monkeypatch.setattr(builtins, "open", observed_open)
    elif kind.startswith("public"):
        target = artifacts._secure_temporary_file
        opname, argval = "UNPACK_SEQUENCE", None
        original_mkstemp = artifacts.tempfile.mkstemp

        def observed_mkstemp(*args, **kwargs):
            descriptor, name = original_mkstemp(*args, **kwargs)
            observed.append((descriptor, Path(name)))
            return descriptor, name

        monkeypatch.setattr(artifacts.tempfile, "mkstemp", observed_mkstemp)
    else:
        target = artifacts._secure_temporary_file_at
        opname, argval = "STORE_FAST", "descriptor"
        original_open = artifacts.os.open

        def observed_open(path, flags, *args, **kwargs):
            descriptor = original_open(path, flags, *args, **kwargs)
            if kwargs.get("dir_fd") == directory_fd:
                observed.append((descriptor, root / os.fspath(path)))
            return descriptor

        monkeypatch.setattr(artifacts.os, "open", observed_open)

    try:
        if hasattr(artifacts, "_staged_file"):
            def opened_handle_is_not_owned(frame, instruction):
                if (
                    instruction.opname != "STORE_FAST"
                    or instruction.argval != "handle"
                    or frame.f_locals.get("handle") is not None
                ):
                    return False
                for descriptor, path in observed:
                    try:
                        os.fstat(descriptor)
                    except OSError:
                        continue
                    if path.exists():
                        return True
                return False

            _interrupt_at_instruction(
                target, invoke, opened_handle_is_not_owned
            )
        else:
            _interrupt_at_opcode(target, opname, argval, invoke)
        leaked = []
        for descriptor, _path in observed:
            try:
                os.fstat(descriptor)
            except OSError:
                continue
            leaked.append(descriptor)
        leftovers = _temporary_entries(root)
    finally:
        monkeypatch.undo()
        for descriptor, path in observed:
            try:
                original_close(descriptor)
            except OSError:
                pass
            path.unlink(missing_ok=True)
        if directory_fd is not None:
            original_close(directory_fd)

    assert observed
    assert leaked == []
    assert leftovers == []


@pytest.mark.parametrize(
    "kind",
    ["public_json", "public_json_create", "public_torch", "at_json", "at_torch"],
)
def test_staging_file_adoption_interrupt_never_recloses_a_reused_fd(
    tmp_path,
    monkeypatch,
    kind,
):
    root, directory_fd, invoke = _publication_case(tmp_path, kind)
    original_close = os.close
    original_open = os.open
    resource_fd = None
    replacement_fd = None

    if hasattr(artifacts, "_staged_file"):
        target = artifacts._staged_file
        original_builtin_open = builtins.open

        def observed_builtin_open(path, mode="r", *args, **kwargs):
            nonlocal resource_fd
            handle = original_builtin_open(path, mode, *args, **kwargs)
            if mode == "xb" and os.fspath(path).endswith(".tmp"):
                resource_fd = handle.fileno()
            return handle

        monkeypatch.setattr(builtins, "open", observed_builtin_open)
    else:
        target = {
            "public_json": artifacts.atomic_json,
            "public_torch": artifacts.atomic_torch,
            "at_json": artifacts._atomic_json_at,
            "at_torch": artifacts._atomic_torch_at,
        }[kind]
        original_fdopen = artifacts.os.fdopen

        def observed_fdopen(descriptor, *args, **kwargs):
            nonlocal resource_fd
            handle = original_fdopen(descriptor, *args, **kwargs)
            resource_fd = handle.fileno()
            return handle

        monkeypatch.setattr(artifacts.os, "fdopen", observed_fdopen)

    def detect_stale_close(descriptor):
        nonlocal replacement_fd
        if descriptor == resource_fd:
            try:
                os.fstat(descriptor)
            except OSError:
                replacement_fd = original_open(os.devnull, os.O_RDONLY)
        return original_close(descriptor)

    monkeypatch.setattr(artifacts.os, "close", detect_stale_close)
    try:
        if hasattr(artifacts, "_staged_file"):
            _interrupt_at_instruction(
                target,
                invoke,
                lambda frame, instruction: (
                    instruction.opname == "STORE_FAST"
                    and instruction.argval == "handle"
                    and frame.f_locals.get("handle") is None
                    and resource_fd is not None
                ),
            )
        else:
            _interrupt_at_opcode(target, "STORE_FAST", "handle", invoke)
        replacement_survived = True
        if replacement_fd is not None:
            try:
                os.fstat(replacement_fd)
            except OSError:
                replacement_survived = False
        leftovers = _temporary_entries(root)
    finally:
        monkeypatch.undo()
        if replacement_fd is not None:
            try:
                original_close(replacement_fd)
            except OSError:
                pass
        if directory_fd is not None:
            original_close(directory_fd)
        for path in _temporary_entries(root):
            path.unlink()

    assert resource_fd is not None
    assert replacement_survived
    assert leftovers == []


def test_staged_file_yield_interrupt_cleans_the_unique_entry(tmp_path):
    target = tmp_path / "value.json"
    assert hasattr(artifacts, "_staged_file")

    _interrupt_at_opcode(
        artifacts._staged_file,
        "YIELD_VALUE",
        None,
        lambda: artifacts.atomic_json({"value": 1}, target),
    )

    assert _temporary_entries(tmp_path) == []
    assert not target.exists()


@pytest.mark.parametrize(
    "kind",
    ["public_json", "public_json_create", "public_torch", "at_json", "at_torch"],
)
def test_staged_context_enter_return_interrupt_cleans_owned_file(
    tmp_path,
    monkeypatch,
    kind,
):
    root, directory_fd, invoke = _publication_case(tmp_path, kind)
    target = {
        "public_json": artifacts.atomic_json,
        "public_json_create": artifacts._atomic_json_create,
        "public_torch": artifacts.atomic_torch,
        "at_json": artifacts._atomic_json_at,
        "at_torch": artifacts._atomic_torch_at,
    }[kind]
    original_open = builtins.open
    observed = []

    def observed_open(path, mode="r", *args, **kwargs):
        handle = original_open(path, mode, *args, **kwargs)
        if mode == "xb" and os.fspath(path).endswith(".tmp"):
            observed.append(handle.fileno())
        return handle

    monkeypatch.setattr(builtins, "open", observed_open)
    try:
        _interrupt_at_opcode(target, "UNPACK_SEQUENCE", None, invoke)
        leaked = []
        for descriptor in observed:
            try:
                os.fstat(descriptor)
            except OSError:
                continue
            leaked.append(descriptor)
        leftovers = _temporary_entries(root)
    finally:
        monkeypatch.undo()
        if directory_fd is not None:
            os.close(directory_fd)

    assert observed
    assert leaked == []
    assert leftovers == []


@pytest.mark.parametrize("kind", ["public_json", "at_json"])
def test_staged_file_close_completion_never_recloses_a_reused_fd(
    tmp_path,
    monkeypatch,
    kind,
):
    root, directory_fd, invoke = _publication_case(tmp_path, kind)
    function = _traceable(artifacts._staged_file)
    close_returns = _attribute_call_return_offsets(function, "close")
    original_builtin_open = builtins.open
    resource_fd = None
    replacement_fd = None

    def observed_open(path, mode="r", *args, **kwargs):
        nonlocal resource_fd
        handle = original_builtin_open(path, mode, *args, **kwargs)
        if mode == "xb" and os.fspath(path).endswith(".tmp"):
            resource_fd = handle.fileno()
        return handle

    def closed_resource(_frame, instruction):
        if instruction.offset not in close_returns or resource_fd is None:
            return False
        try:
            os.fstat(resource_fd)
        except OSError:
            return True
        return False

    def reuse_closed_descriptor(_frame, _instruction):
        nonlocal replacement_fd
        replacement_fd = os.open(os.devnull, os.O_RDONLY)
        assert replacement_fd == resource_fd

    monkeypatch.setattr(builtins, "open", observed_open)
    try:
        _interrupt_at_instruction(
            function,
            invoke,
            closed_resource,
            before_interrupt=reuse_closed_descriptor,
        )
    finally:
        monkeypatch.undo()
        if directory_fd is not None:
            os.close(directory_fd)

    assert replacement_fd is not None
    os.fstat(replacement_fd)
    os.close(replacement_fd)
    assert _temporary_entries(root) == []


@pytest.mark.parametrize(
    "kind", ["read_json", "sha256_at", "shard_path", "shard_at"]
)
def test_read_open_return_interrupt_has_no_raw_descriptor(
    tmp_path,
    monkeypatch,
    kind,
):
    _root, directory_fd, invoke = _read_case(tmp_path, kind)
    original_close = os.close
    observed = []
    target = artifacts._open_regular_file_object
    original_fileio = artifacts.io.FileIO

    class ObservedFileIO(original_fileio):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            observed.append(self.fileno())

    monkeypatch.setattr(artifacts.io, "FileIO", ObservedFileIO)
    try:
        _interrupt_at_instruction(
            target,
            invoke,
            lambda _frame, instruction: (
                instruction.opname == "RETURN_VALUE"
                and observed
                and Path(f"/proc/self/fd/{observed[-1]}").exists()
            ),
        )
        leaked = []
        for descriptor in observed:
            try:
                os.fstat(descriptor)
            except OSError:
                continue
            leaked.append(descriptor)
    finally:
        monkeypatch.undo()
        for descriptor in observed:
            if descriptor == directory_fd:
                continue
            try:
                original_close(descriptor)
            except OSError:
                pass
        original_close(directory_fd)

    assert observed
    assert leaked == []


@pytest.mark.parametrize(
    "kind", ["read_json", "sha256_at", "shard_path", "shard_at"]
)
def test_read_helper_return_interrupt_closes_owned_file(
    tmp_path,
    monkeypatch,
    kind,
):
    _root, directory_fd, invoke = _read_case(tmp_path, kind)
    target = {
        "read_json": artifacts._read_json_at,
        "sha256_at": artifacts._sha256_file_at,
        "shard_path": artifacts._safe_load_shard,
        "shard_at": artifacts._safe_load_shard,
    }[kind]
    original_fileio = artifacts.io.FileIO
    observed = []

    class ObservedFileIO(original_fileio):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            observed.append(self.fileno())

    monkeypatch.setattr(artifacts.io, "FileIO", ObservedFileIO)
    try:
        _interrupt_at_opcode(target, "BEFORE_WITH", None, invoke)
        leaked = []
        for descriptor in observed:
            try:
                os.fstat(descriptor)
            except OSError:
                continue
            leaked.append(descriptor)
    finally:
        monkeypatch.undo()
        os.close(directory_fd)

    assert observed
    assert leaked == []


@pytest.mark.parametrize(
    "kind", ["read_json", "sha256_at", "shard_path", "shard_at"]
)
def test_read_file_adoption_interrupt_never_recloses_a_reused_fd(
    tmp_path,
    monkeypatch,
    kind,
):
    _root, directory_fd, invoke = _read_case(tmp_path, kind)
    original_open = os.open
    original_fileio = artifacts.io.FileIO
    resource_fd = None
    replacement_fd = None

    class ReusingFileIO(original_fileio):
        def __del__(self):
            nonlocal resource_fd, replacement_fd
            if not self.closed:
                resource_fd = self.fileno()
                self.close()
                replacement_fd = original_open(os.devnull, os.O_RDONLY)
                assert replacement_fd == resource_fd

    monkeypatch.setattr(artifacts.io, "FileIO", ReusingFileIO)
    try:
        _interrupt_at_instruction(
            artifacts._open_regular_file_object,
            invoke,
            lambda _frame, instruction: instruction.opname == "RETURN_VALUE",
        )
        assert replacement_fd is not None
        os.fstat(replacement_fd)
    finally:
        monkeypatch.undo()
        if replacement_fd is not None:
            os.close(replacement_fd)
        os.close(directory_fd)

    assert resource_fd is not None


@pytest.mark.parametrize("use_directory_fd", [False, True])
def test_validation_failure_close_completion_never_recloses_a_reused_fd(
    tmp_path,
    monkeypatch,
    use_directory_fd,
):
    root = tmp_path / "cache"
    root.mkdir()
    target = root / "value.bin"
    target.write_bytes(b"value")
    directory_fd = (
        os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        if use_directory_fd
        else None
    )
    source = target if directory_fd is None else target.name
    original_fileio = artifacts.io.FileIO
    original_open = os.open
    original_fstat = artifacts.os.fstat
    resource_fd = None
    replacement_fd = None
    validation_failed = False

    class InterruptingCloseFileIO(original_fileio):
        def __init__(self, *args, **kwargs):
            nonlocal resource_fd
            super().__init__(*args, **kwargs)
            resource_fd = self.fileno()

        def close(self):
            nonlocal resource_fd, replacement_fd
            if not self.closed:
                resource_fd = self.fileno()
                super().close()
                replacement_fd = original_open(os.devnull, os.O_RDONLY)
                assert replacement_fd == resource_fd
                raise KeyboardInterrupt("interrupted after close completed")
            return super().close()

    def fail_validation(descriptor):
        nonlocal validation_failed
        if descriptor == resource_fd and not validation_failed:
            validation_failed = True
            raise RuntimeError("validation failed")
        return original_fstat(descriptor)

    monkeypatch.setattr(artifacts.io, "FileIO", InterruptingCloseFileIO)
    monkeypatch.setattr(artifacts.os, "fstat", fail_validation)
    try:
        with pytest.raises(
            KeyboardInterrupt, match="interrupted after close completed"
        ):
            artifacts._open_regular_file(
                source,
                directory_fd=directory_fd,
                error_message="invalid test file",
            )
    finally:
        monkeypatch.undo()
        if directory_fd is not None:
            os.close(directory_fd)

    assert validation_failed
    assert replacement_fd is not None
    os.fstat(replacement_fd)
    os.close(replacement_fd)


@pytest.mark.parametrize(
    "kind", ["read_json", "sha256_at", "shard_path", "shard_at"]
)
def test_reads_reject_an_entry_replaced_during_handle_validation(
    tmp_path,
    monkeypatch,
    kind,
):
    root, directory_fd, invoke = _read_case(tmp_path, kind)
    name = {
        "read_json": "value.json",
        "sha256_at": "value.bin",
        "shard_path": "shard_00000.pt",
        "shard_at": "shard_00000.pt",
    }[kind]
    replacement = tmp_path / f"{kind}.replacement"
    if kind == "read_json":
        replacement.write_text('{"value": 2}\n', encoding="utf-8")
    elif kind == "sha256_at":
        replacement.write_bytes(b"replacement")
    else:
        torch.save([_record("replacement")], replacement)
    original_fstat = artifacts.os.fstat
    swapped = False

    def swap_before_validation(descriptor):
        nonlocal swapped
        state = original_fstat(descriptor)
        if (
            not swapped
            and descriptor != directory_fd
            and artifacts.stat.S_ISREG(state.st_mode)
        ):
            (root / name).replace(root / f"{name}.original")
            replacement.replace(root / name)
            swapped = True
        return state

    monkeypatch.setattr(artifacts.os, "fstat", swap_before_validation)
    try:
        with pytest.raises(ValueError, match="changed|regular file"):
            invoke()
    finally:
        monkeypatch.undo()
        os.close(directory_fd)

    assert swapped


@pytest.mark.parametrize("kind", ["read_json", "sha256_at"])
def test_dirfd_reads_reject_symlink_entries(tmp_path, kind):
    root = tmp_path / kind
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text('{"value": 1}\n', encoding="utf-8")
    name = "value.json" if kind == "read_json" else "value.bin"
    (root / name).symlink_to(outside)
    directory_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        with pytest.raises(ValueError, match="regular file"):
            if kind == "read_json":
                artifacts._read_json_at(directory_fd, name)
            else:
                artifacts._sha256_file_at(directory_fd, name)
    finally:
        os.close(directory_fd)


@pytest.mark.parametrize("kind", ["read_json", "shard_path", "shard_at"])
def test_reads_reject_symlink_before_opening_its_fifo_target(
    tmp_path,
    monkeypatch,
    kind,
):
    root = tmp_path / kind
    root.mkdir()
    fifo = tmp_path / "blocking-fifo"
    os.mkfifo(fifo)
    name = "value.json" if kind == "read_json" else "shard_00000.pt"
    (root / name).symlink_to(fifo)
    directory_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    if kind == "read_json":
        invoke = lambda: artifacts._read_json_at(directory_fd, name)
    else:
        invoke = lambda: artifacts._safe_load_shard(
            root,
            name,
            "0" * 64,
            directory_fd=directory_fd if kind == "shard_at" else None,
        )
    original_open = builtins.open

    def forbid_fifo_open(path, mode="r", *args, **kwargs):
        if mode == "rb" and os.fspath(path).endswith(name):
            raise AssertionError("reader followed the symlink before validation")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", forbid_fifo_open)
    try:
        with pytest.raises(ValueError, match="regular file"):
            invoke()
    finally:
        os.close(directory_fd)


@pytest.mark.parametrize(
    "operation",
    [
        "manifest_symlink",
        "provenance_symlink",
        "manifest_swap",
        "manifest_fifo_swap",
    ],
)
def test_fixed_public_json_reads_reject_fifo_entries_without_blocking(
    tmp_path,
    operation,
):
    root = tmp_path / operation
    root.mkdir()
    fifo = tmp_path / f"{operation}.fifo"
    os.mkfifo(fifo)
    if operation == "manifest_symlink":
        (root / "manifest.json").symlink_to(fifo)
    elif operation == "provenance_symlink":
        artifacts_directory = root / "artifacts"
        artifacts_directory.mkdir()
        (artifacts_directory / "provenance.json").symlink_to(fifo)
    else:
        atomic_json(
            {
                "schema_version": 1,
                "record_count": 0,
                "shards": [],
                "shard_sha256": {},
            },
            root / "manifest.json",
        )

    project_root = Path(artifacts.__file__).resolve().parent.parent
    script = r"""
import sys
from pathlib import Path
import differential_uncertainty.artifacts as artifacts

operation = sys.argv[1]
root = Path(sys.argv[2])
fifo = Path(sys.argv[3])
if operation in ("manifest_swap", "manifest_fifo_swap"):
    manifest = root / "manifest.json"
    original = artifacts._stat_entry
    swapped = False

    def swap_after_prevalidation(path, *, directory_fd):
        global swapped
        state = original(path, directory_fd=directory_fd)
        if not swapped and Path(path).name == manifest.name:
            manifest.rename(root / "manifest.original")
            if operation == "manifest_fifo_swap":
                fifo.rename(manifest)
            else:
                manifest.symlink_to(fifo)
            swapped = True
        return state

    artifacts._stat_entry = swap_after_prevalidation

try:
    if operation == "provenance_symlink":
        artifacts.ensure_provenance(root, {"schema_version": 1})
    else:
        artifacts.load_manifest(root)
except ValueError as error:
    if operation == "manifest_swap":
        assert str(error) == "artifact manifest must be a regular file"
    raise SystemExit(0)
raise SystemExit(3)
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            os.fspath(project_root),
            environment.get("PYTHONPATH", ""),
        ]
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            operation,
            os.fspath(root),
            os.fspath(fifo),
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_dirfd_torch_save_failure_closes_and_removes_staging_file(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    root.mkdir()
    directory_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))

    def fail_save(_value, handle):
        handle.write(b"partial")
        raise RuntimeError("save failed")

    monkeypatch.setattr(artifacts.torch, "save", fail_save)
    before = artifacts._live_fd_snapshot()
    try:
        with pytest.raises(RuntimeError, match="save failed"):
            artifacts._atomic_torch_at({"value": 1}, "value.pt", directory_fd)
        after = artifacts._live_fd_snapshot()
        leftovers = _temporary_entries(root)
    finally:
        os.close(directory_fd)

    assert after == before
    assert leftovers == []


@pytest.mark.parametrize(
    "kind", ["public_json", "public_json_create", "at_json"]
)
def test_json_write_failure_closes_and_removes_staging_file(
    tmp_path,
    monkeypatch,
    kind,
):
    root, directory_fd, invoke = _publication_case(tmp_path, kind)
    target = root / "value.json"
    target.write_bytes(b"previous")
    original_open = builtins.open

    class FailingWriteHandle:
        def __init__(self, handle):
            self._handle = handle

        def __getattr__(self, name):
            return getattr(self._handle, name)

        def write(self, value):
            self._handle.write(value[:1])
            raise RuntimeError("write failed")

    def failing_open(path, mode="r", *args, **kwargs):
        handle = original_open(path, mode, *args, **kwargs)
        if mode == "xb":
            return FailingWriteHandle(handle)
        return handle

    monkeypatch.setattr(builtins, "open", failing_open)
    before = artifacts._live_fd_snapshot()
    try:
        with pytest.raises(RuntimeError, match="write failed"):
            invoke()
        after = artifacts._live_fd_snapshot()
        leftovers = _temporary_entries(root)
    finally:
        monkeypatch.undo()
        if directory_fd is not None:
            os.close(directory_fd)

    assert after == before
    assert target.read_bytes() == b"previous"
    assert leftovers == []


@pytest.mark.parametrize("kind", ["public_json", "at_json"])
def test_staged_file_setup_failure_closes_and_removes_entry(
    tmp_path,
    monkeypatch,
    kind,
):
    root, directory_fd, invoke = _publication_case(tmp_path, kind)
    monkeypatch.setattr(
        artifacts.os,
        "fchmod",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("setup failed")),
    )
    before = artifacts._live_fd_snapshot()
    try:
        with pytest.raises(RuntimeError, match="setup failed"):
            invoke()
        after = artifacts._live_fd_snapshot()
        leftovers = _temporary_entries(root)
    finally:
        if directory_fd is not None:
            os.close(directory_fd)

    assert after == before
    assert leftovers == []


def test_directory_fsync_open_return_interrupt_has_no_raw_descriptor(tmp_path):
    (tmp_path / "published-value").write_bytes(b"value")
    before = artifacts._live_fd_snapshot()
    _interrupt_at_opcode(
        artifacts._fsync_directory,
        "STORE_FAST",
        "descriptor",
        lambda: artifacts._fsync_directory(tmp_path),
    )
    after = artifacts._live_fd_snapshot()
    leaked = after - before
    for descriptor in leaked:
        os.close(descriptor)

    assert leaked == set()


@pytest.mark.parametrize(
    "kind",
    ["public_json", "public_json_create", "public_torch", "at_json", "at_torch"],
)
def test_publication_rejects_replaced_staging_entry(
    tmp_path,
    monkeypatch,
    kind,
):
    root, directory_fd, invoke = _publication_case(tmp_path, kind)
    target = root / ("value.pt" if "torch" in kind else "value.json")
    retained = root / "retained-correct-staging"
    original_fsync = artifacts.os.fsync
    swapped = False

    def swap_after_file_fsync(descriptor):
        nonlocal swapped
        result = original_fsync(descriptor)
        state = os.fstat(descriptor)
        if not swapped and artifacts.stat.S_ISREG(state.st_mode):
            staging = _temporary_entries(root)
            assert len(staging) == 1
            staging[0].replace(retained)
            if "torch" in kind:
                torch.save({"attacker": True}, staging[0])
            else:
                staging[0].write_text(
                    '{"attacker": true}\n', encoding="utf-8"
                )
            swapped = True
        return result

    monkeypatch.setattr(artifacts.os, "fsync", swap_after_file_fsync)
    try:
        with pytest.raises(ValueError, match="staging entry changed"):
            invoke()
    finally:
        if directory_fd is not None:
            os.close(directory_fd)

    assert swapped
    assert retained.exists()
    assert not target.exists()
    assert _temporary_entries(root) == []


@pytest.mark.parametrize("kind", ["read_json", "shard_path", "shard_at"])
def test_read_or_load_failure_closes_the_owned_file(
    tmp_path,
    monkeypatch,
    kind,
):
    _root, directory_fd, invoke = _read_case(tmp_path, kind)
    if kind == "read_json":
        monkeypatch.setattr(
            artifacts.json,
            "load",
            lambda _handle: (_ for _ in ()).throw(RuntimeError("load failed")),
        )
    else:
        monkeypatch.setattr(
            artifacts.torch,
            "load",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("load failed")
            ),
        )
    before = artifacts._live_fd_snapshot()
    try:
        with pytest.raises(RuntimeError, match="load failed"):
            invoke()
        after = artifacts._live_fd_snapshot()
    finally:
        os.close(directory_fd)

    assert after == before


def test_atomic_json_and_torch_publish_complete_values_without_staging_files(tmp_path):
    json_path = tmp_path / "nested" / "value.json"
    tensor_path = tmp_path / "nested" / "value.pt"

    atomic_json({"z": 2, "a": [1, 3]}, json_path)
    atomic_torch({"value": torch.tensor([4, 5])}, tensor_path)

    assert json.loads(json_path.read_text(encoding="utf-8")) == {
        "a": [1, 3],
        "z": 2,
    }
    loaded = torch.load(tensor_path, map_location="cpu", weights_only=True)
    torch.testing.assert_close(loaded["value"], torch.tensor([4, 5]))
    assert list(tmp_path.rglob("*.tmp")) == []
    assert not (json_path.parent / artifacts._DIRECTORY_ANCHOR).exists()


def test_failed_atomic_torch_publication_preserves_the_target(tmp_path, monkeypatch):
    target = tmp_path / "value.pt"
    target.write_bytes(b"previous")

    def fail_after_staging(_value, destination):
        if hasattr(destination, "write"):
            destination.write(b"incomplete")
        else:
            Path(destination).write_bytes(b"incomplete")
        raise RuntimeError("save failed")

    monkeypatch.setattr(artifacts.torch, "save", fail_after_staging)

    with pytest.raises(RuntimeError, match="save failed"):
        atomic_torch({"new": True}, target)

    assert target.read_bytes() == b"previous"
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_json_does_not_follow_a_predictable_staging_symlink(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("protected", encoding="utf-8")
    target = tmp_path / "value.json"
    predictable = tmp_path / "value.json.tmp"
    predictable.symlink_to(victim)

    atomic_json({"value": 3}, target)

    assert victim.read_text(encoding="utf-8") == "protected"
    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 3}
    assert predictable.is_symlink()


def test_atomic_torch_does_not_follow_a_predictable_staging_symlink(tmp_path):
    victim = tmp_path / "victim.bin"
    victim.write_bytes(b"protected")
    target = tmp_path / "value.pt"
    predictable = tmp_path / "value.pt.tmp"
    predictable.symlink_to(victim)

    atomic_torch({"value": torch.tensor([3])}, target)

    assert victim.read_bytes() == b"protected"
    loaded = torch.load(target, map_location="cpu", weights_only=True)
    torch.testing.assert_close(loaded["value"], torch.tensor([3]))
    assert predictable.is_symlink()


def test_atomic_publication_fsyncs_staged_files_and_parent_directories(
    tmp_path,
    monkeypatch,
):
    fsynced = []
    monkeypatch.setattr(artifacts.os, "fsync", fsynced.append)

    atomic_json({"value": 1}, tmp_path / "value.json")
    atomic_torch({"value": torch.tensor([2])}, tmp_path / "value.pt")

    assert len(fsynced) == 4


def test_sha256_file_hashes_content_in_chunks_not_the_filename(tmp_path):
    first = tmp_path / "a.bin"
    second = tmp_path / "b.bin"
    first.write_bytes(b"same content")
    second.write_bytes(b"same content")

    expected = hashlib.sha256(b"same content").hexdigest()
    assert sha256_file(first, chunk_size=2) == expected
    assert sha256_file(second, chunk_size=3) == expected


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True])
def test_sha256_file_rejects_invalid_chunk_sizes(tmp_path, chunk_size):
    path = tmp_path / "value.bin"
    path.write_bytes(b"value")

    with pytest.raises(ValueError, match="chunk_size"):
        sha256_file(path, chunk_size=chunk_size)


def test_source_digest_is_clone_stable_and_depends_on_relative_names_and_content(
    tmp_path,
):
    first_root = tmp_path / "first-clone"
    second_root = tmp_path / "second-clone"
    for root in (first_root, second_root):
        (root / "package").mkdir(parents=True)
        (root / "package" / "a.py").write_text("A = 1\n", encoding="utf-8")
        (root / "package" / "b.py").write_text("B = 2\n", encoding="utf-8")

    first = source_digest(
        [first_root / "package" / "b.py", first_root / "package" / "a.py"],
        root=first_root,
    )
    second = source_digest(
        [second_root / "package" / "a.py", second_root / "package" / "b.py"],
        root=second_root,
    )
    assert first == second

    (second_root / "package" / "b.py").write_text("B = 3\n", encoding="utf-8")
    changed_content = source_digest(
        [second_root / "package" / "a.py", second_root / "package" / "b.py"],
        root=second_root,
    )
    assert changed_content != first

    (second_root / "renamed.py").write_text("A = 1\n", encoding="utf-8")
    changed_name = source_digest([second_root / "renamed.py"], root=second_root)
    original_name = source_digest([second_root / "package" / "a.py"], root=second_root)
    assert changed_name != original_name


def test_source_digest_frames_names_and_content_and_rejects_files_outside_root(
    tmp_path,
):
    root = tmp_path / "root"
    root.mkdir()
    short_name = root / "a"
    long_name = root / "ab"
    short_name.write_bytes(b"bc")
    long_name.write_bytes(b"c")

    assert source_digest([short_name], root=root) != source_digest(
        [long_name], root=root
    )
    with pytest.raises(ValueError, match="outside source root"):
        source_digest([tmp_path / "outside.py"], root=root)


def test_provenance_is_published_once_and_accepts_a_json_equivalent_repeat(tmp_path):
    run = tmp_path / "run"
    expected = {
        "schema_version": 1,
        "checkpoint_sha256": "first",
        "config": {"layers": (2,), "radii": {0: 0.0, 10: 12.0}},
    }

    path = ensure_provenance(run, expected)
    ensure_provenance(
        run,
        {
            "schema_version": 1,
            "checkpoint_sha256": "first",
            "config": {"layers": [2], "radii": {"0": 0.0, "10": 12.0}},
        },
    )

    assert path == run / "artifacts" / "provenance.json"
    assert json.loads(path.read_text(encoding="utf-8"))["checkpoint_sha256"] == "first"
    assert list(run.rglob("*.tmp")) == []


def test_concurrent_provenance_publishers_cannot_both_accept_different_runs(
    tmp_path,
    monkeypatch,
):
    run = tmp_path / "run"
    first = {"schema_version": 1, "checkpoint_sha256": "first"}
    second = {"schema_version": 1, "checkpoint_sha256": "second"}
    atomic_barrier = threading.Barrier(2)
    link_barrier = threading.Barrier(2)
    original_atomic_json = artifacts.atomic_json
    original_link = artifacts.os.link

    def synchronized_atomic_json(value, path):
        if Path(path).name == "provenance.json":
            atomic_barrier.wait(timeout=5)
        return original_atomic_json(value, path)

    def synchronized_link(*args, **kwargs):
        link_barrier.wait(timeout=5)
        return original_link(*args, **kwargs)

    monkeypatch.setattr(artifacts, "atomic_json", synchronized_atomic_json)
    monkeypatch.setattr(artifacts.os, "link", synchronized_link)
    outcomes = []

    def publish(expected):
        try:
            ensure_provenance(run, expected)
        except Exception as error:
            outcomes.append((type(error).__name__, expected["checkpoint_sha256"]))
        else:
            outcomes.append(("ok", expected["checkpoint_sha256"]))

    threads = [
        threading.Thread(target=publish, args=(expected,))
        for expected in (first, second)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not any(thread.is_alive() for thread in threads)
    assert sorted(status for status, _checkpoint in outcomes) == ["ValueError", "ok"]
    winner = next(checkpoint for status, checkpoint in outcomes if status == "ok")
    persisted = json.loads(
        (run / "artifacts" / "provenance.json").read_text(encoding="utf-8")
    )
    assert persisted["checkpoint_sha256"] == winner


def test_failed_exclusive_provenance_link_cleans_its_staging_file(
    tmp_path,
    monkeypatch,
):
    run = tmp_path / "run"

    def fail_link(*_args, **_kwargs):
        raise OSError("link failed")

    monkeypatch.setattr(artifacts.os, "link", fail_link)
    with pytest.raises(OSError, match="link failed"):
        ensure_provenance(run, {"schema_version": 1})

    assert not (run / "artifacts" / "provenance.json").exists()
    assert list(run.rglob("*.tmp")) == []


@pytest.mark.parametrize(
    "changed, mismatched_key",
    [
        (
            {"schema_version": 1, "checkpoint_sha256": "second", "config": {"k": 5}},
            "checkpoint_sha256",
        ),
        ({"schema_version": 1, "checkpoint_sha256": "first"}, "config"),
        (
            {
                "schema_version": 1,
                "checkpoint_sha256": "first",
                "config": {"k": 5},
                "extra": True,
            },
            "extra",
        ),
    ],
)
def test_provenance_refuses_any_exact_key_or_value_mismatch(
    tmp_path,
    changed,
    mismatched_key,
):
    run = tmp_path / "run"
    original = {
        "schema_version": 1,
        "checkpoint_sha256": "first",
        "config": {"k": 5},
    }
    ensure_provenance(run, original)

    with pytest.raises(ValueError, match=mismatched_key):
        ensure_provenance(run, changed)


def test_completed_shards_publish_a_final_manifest_and_iterate_in_order(tmp_path):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation", "input_id": "abc"}

    with ShardWriter(root, metadata, shard_size=2) as writer:
        writer.add(_record("a", severity=0, value=1))
        writer.add(_record("a", severity=1, value=2))
        writer.add(_record("b", severity=0, value=3))

    manifest = load_manifest(root)
    assert manifest == {
        "input_id": "abc",
        "record_count": 3,
        "schema_version": 1,
        "shards": ["shard_00000.pt", "shard_00001.pt"],
        "shard_sha256": {
            "shard_00000.pt": sha256_file(root / "shard_00000.pt"),
            "shard_00001.pt": sha256_file(root / "shard_00001.pt"),
        },
        "stage": "evaluation",
    }
    assert [
        (record["image_id"], record["severity"], record["value"].item())
        for record in iter_records(root)
    ] == [("a", 0, 1), ("a", 1, 2), ("b", 0, 3)]
    assert sorted(path.name for path in root.iterdir()) == [
        ".artifact_directory_anchor",
        "manifest.json",
        "shard_00000.pt",
        "shard_00001.pt",
    ]


def test_partial_shards_resume_from_published_record_keys(tmp_path):
    root = tmp_path / "cache"
    metadata = {"stage": "reference", "input_id": "abc"}
    writer = ShardWriter(root, metadata, shard_size=1)
    writer.add(_record("a"))

    resumed = ShardWriter(root, metadata, shard_size=1)
    assert resumed.existing_keys() == {("a", 0)}
    resumed.add(_record("b", value=2))
    resumed.close()

    assert [record["image_id"] for record in iter_records(root)] == ["a", "b"]


def test_exceptional_context_keeps_only_full_shards_resumable(tmp_path):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation"}

    with pytest.raises(RuntimeError, match="interrupted"):
        with ShardWriter(root, metadata, shard_size=2) as writer:
            writer.add(_record("a"))
            writer.add(_record("b"))
            writer.add(_record("not-published"))
            raise RuntimeError("interrupted")

    assert not (root / "manifest.json").exists()
    resumed = ShardWriter(root, metadata, shard_size=2)
    assert resumed.existing_keys() == {("a", 0), ("b", 0)}
    resumed.close()
    assert [record["image_id"] for record in iter_records(root)] == ["a", "b"]


def test_same_process_handoff_invalidates_the_predecessor(tmp_path):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation"}
    writer = ShardWriter(root, metadata, shard_size=1)
    writer.add(_record("a"))

    resumed = ShardWriter(root, metadata, shard_size=1)

    with pytest.raises(RuntimeError, match="superseded"):
        writer.add(_record("stale-add"))
    with pytest.raises(RuntimeError, match="superseded"):
        writer._flush()
    with pytest.raises(RuntimeError, match="superseded"):
        writer._publish_partial()
    writer.close()
    assert not (root / "manifest.json").exists()

    resumed.add(_record("b"))
    resumed.close()
    assert [record["image_id"] for record in iter_records(root)] == ["a", "b"]


def test_handoff_refuses_unpublished_buffer_without_invalidating_writer(tmp_path):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation"}
    writer = ShardWriter(root, metadata, shard_size=2)
    writer.add(_record("a"))
    partial_before = (root / "partial_manifest.json").read_bytes()

    with pytest.raises(RuntimeError, match="unpublished buffer"):
        ShardWriter(root, metadata, shard_size=2)

    assert (root / "partial_manifest.json").read_bytes() == partial_before
    writer.add(_record("b"))
    resumed = ShardWriter(root, metadata, shard_size=2)
    resumed.close()
    assert [record["image_id"] for record in iter_records(root)] == ["a", "b"]


def test_failed_handoff_metadata_check_leaves_predecessor_active(tmp_path):
    root = tmp_path / "cache"
    metadata = {"stage": "reference", "input_id": "abc"}
    writer = ShardWriter(root, metadata, shard_size=1)
    writer.add(_record("a"))

    with pytest.raises(ValueError, match="input_id"):
        ShardWriter(
            root,
            {"stage": "reference", "input_id": "different"},
            shard_size=1,
        )

    writer.add(_record("b"))
    writer.close()
    assert [record["image_id"] for record in iter_records(root)] == ["a", "b"]


def test_concurrent_handoff_and_close_do_not_deadlock_or_publish_twice(tmp_path):
    for index in range(20):
        root = tmp_path / f"cache-{index}"
        metadata = {"stage": "evaluation"}
        writer = ShardWriter(root, metadata, shard_size=1)
        writer.add(_record("a"))
        barrier = threading.Barrier(2)
        successors = []
        errors = []

        def close_predecessor():
            try:
                barrier.wait()
                writer.close()
            except BaseException as error:
                errors.append(error)

        def attempt_handoff():
            try:
                barrier.wait()
                successors.append(
                    ShardWriter(root, metadata, shard_size=1)
                )
            except FileExistsError:
                pass
            except BaseException as error:
                errors.append(error)

        threads = [
            threading.Thread(target=close_predecessor),
            threading.Thread(target=attempt_handoff),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        assert not any(thread.is_alive() for thread in threads)
        assert errors == []
        for successor in successors:
            successor.close()
        assert [record["image_id"] for record in iter_records(root)] == ["a"]


def test_post_transfer_constructor_failure_releases_ownership(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation"}
    writer = ShardWriter(root, metadata, shard_size=1)
    writer.add(_record("a"))
    original_resume = ShardWriter._resume

    def fail_resume(_self, _state):
        raise RuntimeError("resume failed")

    monkeypatch.setattr(ShardWriter, "_resume", fail_resume)
    with pytest.raises(RuntimeError, match="resume failed"):
        ShardWriter(root, metadata, shard_size=1)
    monkeypatch.setattr(ShardWriter, "_resume", original_resume)

    with pytest.raises(RuntimeError, match="superseded"):
        writer.add(_record("stale"))

    resumed = ShardWriter(root, metadata, shard_size=1)
    resumed.close()
    assert [record["image_id"] for record in iter_records(root)] == ["a"]


def test_handoff_uses_directory_identity_across_symlink_aliases(tmp_path):
    root = tmp_path / "cache"
    alias = tmp_path / "cache-alias"
    metadata = {"stage": "evaluation"}
    writer = ShardWriter(root, metadata, shard_size=1)
    writer.add(_record("a"))
    alias.symlink_to(root, target_is_directory=True)

    resumed = ShardWriter(alias, metadata, shard_size=1)

    with pytest.raises(RuntimeError, match="superseded"):
        writer.add(_record("stale"))
    resumed.close()

    assert [record["image_id"] for record in iter_records(root)] == ["a"]


def test_writer_io_stays_bound_to_the_locked_directory_after_rename(tmp_path):
    root = tmp_path / "cache"
    moved = tmp_path / "moved-cache"
    metadata = {"stage": "evaluation"}
    writer = ShardWriter(root, metadata, shard_size=1)
    root.rename(moved)
    replacement = ShardWriter(root, metadata, shard_size=1)

    writer.add(_record("old-directory"))
    replacement.add(_record("new-directory"))
    writer.close()
    replacement.close()

    assert [record["image_id"] for record in iter_records(moved)] == [
        "old-directory"
    ]
    assert [record["image_id"] for record in iter_records(root)] == [
        "new-directory"
    ]


def test_writer_identity_is_derived_from_the_same_directory_fd_it_locks(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    moved = tmp_path / "moved-cache"
    metadata = {"stage": "evaluation"}
    original_acquire = artifacts._acquire_directory_lock
    replaced = False

    def replace_path_before_acquire(directory, **kwargs):
        nonlocal replaced
        if not replaced:
            Path(directory).rename(moved)
            Path(directory).mkdir()
            replaced = True
        return original_acquire(directory, **kwargs)

    monkeypatch.setattr(
        artifacts,
        "_acquire_directory_lock",
        replace_path_before_acquire,
    )
    writer = ShardWriter(root, metadata, shard_size=1)
    writer.add(_record("a"))
    monkeypatch.setattr(
        artifacts,
        "_acquire_directory_lock",
        original_acquire,
    )

    resumed = ShardWriter(moved, metadata, shard_size=1)
    resumed.add(_record("b"))
    resumed.close()
    replacement = ShardWriter(root, metadata, shard_size=1)
    replacement.add(_record("replacement"))
    replacement.close()

    assert [record["image_id"] for record in iter_records(moved)] == ["a", "b"]
    assert [record["image_id"] for record in iter_records(root)] == [
        "replacement"
    ]


def test_directory_open_is_serialized_with_the_fork_registry(tmp_path, monkeypatch):
    class TrackingRLock:
        def __init__(self):
            self._lock = threading.RLock()
            self._local = threading.local()

        def acquire(self, *args, **kwargs):
            acquired = self._lock.acquire(*args, **kwargs)
            if acquired:
                self._local.depth = getattr(self._local, "depth", 0) + 1
            return acquired

        def release(self):
            self._local.depth -= 1
            self._lock.release()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, *_exc):
            self.release()

        def held_by_current_thread(self):
            return getattr(self._local, "depth", 0) > 0

    tracking_lock = TrackingRLock()
    original_open_directory = artifacts._open_directory
    observed = []

    def observe_registry_boundary(directory, **kwargs):
        observed.append(tracking_lock.held_by_current_thread())
        return original_open_directory(directory, **kwargs)

    monkeypatch.setattr(
        artifacts,
        "_WRITER_REGISTRY_LOCK",
        tracking_lock,
    )
    monkeypatch.setattr(
        artifacts,
        "_open_directory",
        observe_registry_boundary,
    )
    writer = ShardWriter(
        tmp_path / "cache",
        {"stage": "evaluation"},
        shard_size=1,
    )
    writer.close()

    assert observed == [True]


def test_lock_helper_return_exception_releases_caller_owned_fd(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    original_acquire = artifacts._acquire_directory_lock

    def acquire_then_interrupt(directory, **kwargs):
        original_acquire(directory, **kwargs)
        raise KeyboardInterrupt("interrupted after helper return")

    monkeypatch.setattr(
        artifacts,
        "_acquire_directory_lock",
        acquire_then_interrupt,
    )
    try:
        with pytest.raises(KeyboardInterrupt, match="after helper return"):
            ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        assert artifacts._PENDING_LOCK_FDS == set()
        monkeypatch.setattr(
            artifacts,
            "_acquire_directory_lock",
            original_acquire,
        )
        recovered = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        recovered.close()
    finally:
        monkeypatch.setattr(
            artifacts,
            "_acquire_directory_lock",
            original_acquire,
        )
        for descriptor in list(artifacts._PENDING_LOCK_FDS):
            artifacts.fcntl.flock(descriptor, artifacts.fcntl.LOCK_UN)
            os.close(descriptor)
            artifacts._PENDING_LOCK_FDS.discard(descriptor)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_candidate_opened_after_fork_is_cleaned_by_the_child(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation"}
    original_open = artifacts._open_directory
    child_process_id = None
    in_child = False
    release_read, release_write = os.pipe()

    def fork_before_open(directory, **kwargs):
        nonlocal child_process_id, in_child
        process_id = os.fork()
        if process_id == 0:
            in_child = True
            child_process_id = 0
            os.close(release_read)
            return original_open(directory, **kwargs)
        child_process_id = process_id
        os.close(release_write)
        os.read(release_read, 1)
        os.close(release_read)
        return original_open(directory, **kwargs)

    monkeypatch.setattr(artifacts, "_open_directory", fork_before_open)
    try:
        try:
            writer = ShardWriter(root, metadata, shard_size=1)
        except RuntimeError as error:
            if not in_child:
                raise
            monkeypatch.setattr(artifacts, "_open_directory", original_open)
            pending_was_cleared = artifacts._PENDING_LOCK_FDS == set()
            try:
                recovered = ShardWriter(root, metadata, shard_size=1)
            except BaseException:
                recovered_immediately = False
            else:
                recovered_immediately = True
                recovered.__exit__(RuntimeError, None, None)
            os.write(release_write, b"1")
            os.close(release_write)
            os._exit(
                0
                if "fork" in str(error)
                and pending_was_cleared
                and recovered_immediately
                else 50
            )

        if in_child:
            os.write(release_write, b"1")
            os.close(release_write)
            os._exit(51)

        assert child_process_id is not None
        _waited_id, status = os.waitpid(child_process_id, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        writer.close()
    finally:
        if in_child:
            try:
                os.write(release_write, b"1")
            except OSError:
                pass
            try:
                os.close(release_write)
            except OSError:
                pass
            os._exit(52)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_after_candidate_registration_cannot_resurrect_pending_fd(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation"}
    child_process_id = None
    in_child = False
    release_read, release_write = os.pipe()

    class ForkAfterRegistration(set):
        def add(self, lease):
            nonlocal child_process_id, in_child
            super().add(lease)
            process_id = os.fork()
            if process_id == 0:
                in_child = True
                child_process_id = 0
                os.close(release_read)
                return
            child_process_id = process_id
            os.close(release_write)
            os.read(release_read, 1)
            os.close(release_read)

    monkeypatch.setattr(
        artifacts,
        "_PENDING_LOCK_LEASES",
        ForkAfterRegistration(),
    )
    try:
        try:
            writer = ShardWriter(root, metadata, shard_size=1)
        except RuntimeError as error:
            if not in_child:
                raise
            pending_was_cleared = artifacts._PENDING_LOCK_FDS == set()
            try:
                recovered = ShardWriter(root, metadata, shard_size=1)
            except BaseException:
                recovered_immediately = False
            else:
                recovered_immediately = True
                recovered.__exit__(RuntimeError, None, None)
            os.write(release_write, b"1")
            os.close(release_write)
            os._exit(
                0
                if "fork" in str(error)
                and pending_was_cleared
                and recovered_immediately
                else 60
            )

        if in_child:
            os.write(release_write, b"1")
            os.close(release_write)
            os._exit(61)

        assert child_process_id is not None
        _waited_id, status = os.waitpid(child_process_id, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        writer.close()
    finally:
        if in_child:
            try:
                os.write(release_write, b"1")
            except OSError:
                pass
            try:
                os.close(release_write)
            except OSError:
                pass
            os._exit(62)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_parent_continues_when_fork_precedes_candidate_registration(
    tmp_path,
):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation"}
    source_lines, first_line = inspect.getsourcelines(
        artifacts._CandidateLease._adopt_open_descriptor
    )
    registration_line = first_line + next(
        offset
        for offset, line in enumerate(source_lines)
        if line.strip() == "_PENDING_LOCK_LEASES.add(self)"
    )
    child_process_id = None
    in_child = False
    forked = False

    def fork_before_registration(frame, event, _argument):
        nonlocal child_process_id, in_child, forked
        if (
            not forked
            and frame.f_code
            is artifacts._CandidateLease._adopt_open_descriptor.__code__
            and event == "line"
            and frame.f_lineno == registration_line
        ):
            forked = True
            process_id = os.fork()
            if process_id == 0:
                in_child = True
                child_process_id = 0
            else:
                child_process_id = process_id
        return fork_before_registration

    writer = None
    parent_error = None
    try:
        sys.settrace(fork_before_registration)
        try:
            writer = ShardWriter(root, metadata, shard_size=1)
        except RuntimeError as error:
            if in_child:
                pending_was_cleared = (
                    artifacts._PENDING_LOCK_FDS == set()
                    and artifacts._PENDING_LOCK_LEASES == set()
                )
                os._exit(
                    0
                    if "fork" in str(error) and pending_was_cleared
                    else 70
                )
            parent_error = error
        finally:
            sys.settrace(None)

        if in_child:
            os._exit(71)

        assert child_process_id is not None
        _waited_id, status = os.waitpid(child_process_id, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        assert parent_error is None
        assert writer is not None
        writer.close()
    finally:
        sys.settrace(None)
        if in_child:
            os._exit(72)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_shared_raw_descriptor_uses_one_marker_across_fork(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    root.mkdir()
    metadata = {"stage": "evaluation"}
    original_scandir = artifacts.os.scandir
    original_mark = artifacts._mark_lock_descriptor
    child_process_id = None
    in_child = False
    forked = False
    marked = False
    raw_descriptor = None

    def open_then_fork(path):
        nonlocal child_process_id, forked, in_child, raw_descriptor
        before = artifacts._live_fd_snapshot()
        owner = original_scandir(path)
        if Path(path) != root or forked:
            return owner
        candidates = artifacts._live_fd_snapshot() - before
        assert len(candidates) == 1
        forked = True
        raw_descriptor = candidates.pop()
        process_id = os.fork()
        if process_id == 0:
            in_child = True
            child_process_id = 0
        else:
            child_process_id = process_id
        return owner

    def apply_marker(descriptor, marker):
        if marker is None:
            return original_mark(descriptor)
        return original_mark(descriptor, marker)

    def ordered_marker(descriptor, marker=None):
        nonlocal marked
        result = apply_marker(descriptor, marker)
        marked = True
        return result

    monkeypatch.setattr(artifacts.os, "scandir", open_then_fork)
    monkeypatch.setattr(artifacts, "_mark_lock_descriptor", ordered_marker)
    writer = None
    parent_error = None
    try:
        try:
            writer = ShardWriter(root, metadata, shard_size=1)
        except RuntimeError as error:
            if in_child:
                clean_failure = (
                    "fork" in str(error)
                    and artifacts._PENDING_LOCK_FDS == set()
                    and artifacts._PENDING_LOCK_LEASES == set()
                )
                descriptor_was_closed = False
                try:
                    os.fstat(raw_descriptor)
                except OSError as close_error:
                    descriptor_was_closed = (
                        close_error.errno == artifacts.errno.EBADF
                    )
                os._exit(
                    0 if clean_failure and descriptor_was_closed else 80
                )
            parent_error = error

        if in_child:
            os._exit(81)

        assert child_process_id is not None
        _waited_id, status = os.waitpid(child_process_id, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        monkeypatch.setattr(artifacts.os, "scandir", original_scandir)
        monkeypatch.setattr(
            artifacts,
            "_mark_lock_descriptor",
            original_mark,
        )

        assert parent_error is None
        assert writer is not None
        writer.__exit__(RuntimeError, None, None)
        writer = None
        assert raw_descriptor is not None
        with pytest.raises(OSError) as error:
            os.fstat(raw_descriptor)
        assert error.value.errno == artifacts.errno.EBADF
        recovered = ShardWriter(root, metadata, shard_size=1)
        recovered.close()
    finally:
        monkeypatch.setattr(artifacts.os, "scandir", original_scandir)
        monkeypatch.setattr(
            artifacts,
            "_mark_lock_descriptor",
            original_mark,
        )
        if in_child:
            os._exit(82)
        if writer is not None:
            writer.__exit__(RuntimeError, None, None)
        if raw_descriptor is not None:
            try:
                os.close(raw_descriptor)
            except OSError:
                pass


def test_candidate_open_call_return_is_exception_safe(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    original_scandir = artifacts.os.scandir
    opened_descriptor = None
    interrupted = False
    target_line = next(
        instruction.positions.lineno
        for instruction in dis.get_instructions(artifacts._DirectoryHandle.open)
        if instruction.opname == "STORE_ATTR"
        and instruction.argval == "_owner"
    )

    def capture_open(path):
        nonlocal opened_descriptor
        before = artifacts._live_fd_snapshot()
        owner = original_scandir(path)
        if Path(path) == root:
            candidates = artifacts._live_fd_snapshot() - before
            assert len(candidates) == 1
            opened_descriptor = candidates.pop()
        return owner

    def interrupt_after_open(frame, event, _argument):
        nonlocal interrupted
        if frame.f_code is artifacts._DirectoryHandle.open.__code__:
            if (
                not interrupted
                and event == "line"
                and frame.f_lineno == target_line
            ):
                interrupted = True
                raise KeyboardInterrupt("interrupted after directory open")
        return interrupt_after_open

    monkeypatch.setattr(artifacts.os, "scandir", capture_open)
    try:
        sys.settrace(interrupt_after_open)
        with pytest.raises(KeyboardInterrupt, match="after directory open"):
            ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        sys.settrace(None)

        assert interrupted
        assert opened_descriptor is not None
        with pytest.raises(OSError) as error:
            os.fstat(opened_descriptor)
        assert error.value.errno == artifacts.errno.EBADF
        assert artifacts._PENDING_LOCK_FDS == set()
        assert artifacts._PENDING_LOCK_LEASES == set()
        recovered = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        recovered.close()
    finally:
        sys.settrace(None)
        if opened_descriptor is not None:
            try:
                os.close(opened_descriptor)
            except OSError:
                pass


def test_directory_acquisition_avoids_unowned_fwalk_open_result(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"

    def reject_fwalk(*_args, **_kwargs):
        raise AssertionError("directory acquisition must not depend on fwalk")

    monkeypatch.setattr(artifacts.os, "fwalk", reject_fwalk)
    writer = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
    writer.__exit__(RuntimeError, None, None)

    recovered = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
    recovered.close()


def test_directory_acquisition_rejects_concurrent_position_changes_cleanly(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    root.mkdir()
    (root / ".artifact_directory_anchor").mkdir()
    decoy = os.scandir(root)
    original_positions = artifacts._directory_fd_positions
    snapshot_calls = 0

    def advance_decoy_after_first_snapshot(identity):
        nonlocal snapshot_calls
        positions = original_positions(identity)
        snapshot_calls += 1
        if snapshot_calls == 1:
            assert next(decoy).name == ".artifact_directory_anchor"
        return positions

    monkeypatch.setattr(
        artifacts,
        "_directory_fd_positions",
        advance_decoy_after_first_snapshot,
    )
    try:
        with pytest.raises(RuntimeError, match="could not identify"):
            ShardWriter(root, {"stage": "evaluation"}, shard_size=1)

        assert artifacts._PENDING_LOCK_FDS == set()
        assert artifacts._PENDING_LOCK_LEASES == set()
    finally:
        monkeypatch.setattr(
            artifacts,
            "_directory_fd_positions",
            original_positions,
        )
        decoy.close()

    recovered = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
    recovered.close()


def test_scanner_position_binding_survives_descriptor_number_aba(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    root.mkdir()
    (root / ".artifact_directory_anchor").mkdir()
    identity = (os.stat(root).st_dev, os.stat(root).st_ino)
    original_scandir = artifacts.os.scandir
    before = artifacts._live_fd_snapshot()
    stale_owner = original_scandir(root)
    candidates = artifacts._live_fd_snapshot() - before
    stale_fd = next(
        descriptor
        for descriptor in candidates
        if (
            os.fstat(descriptor).st_dev,
            os.fstat(descriptor).st_ino,
        )
        == identity
    )
    decoy_owner = None
    decoy_fd = None
    replaced = False
    writer = None

    def replace_stale_number_and_open_decoy(path):
        nonlocal decoy_fd, decoy_owner, replaced
        if Path(path) != root or replaced:
            return original_scandir(path)
        replaced = True
        stale_owner.close()
        owner = original_scandir(path)
        assert (
            os.fstat(stale_fd).st_dev,
            os.fstat(stale_fd).st_ino,
        ) == identity
        target_snapshot = artifacts._live_fd_snapshot()
        decoy_owner = original_scandir(path)
        new_descriptors = artifacts._live_fd_snapshot() - target_snapshot
        decoy_fd = next(
            descriptor
            for descriptor in new_descriptors
            if (
                os.fstat(descriptor).st_dev,
                os.fstat(descriptor).st_ino,
            )
            == identity
        )
        return owner

    monkeypatch.setattr(
        artifacts.os,
        "scandir",
        replace_stale_number_and_open_decoy,
    )
    try:
        writer = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        assert writer._lock_fd == stale_fd
        assert decoy_fd is not None and decoy_fd != stale_fd
        writer.__exit__(RuntimeError, None, None)
        writer = None
        os.fstat(decoy_fd)
        decoy_owner.close()
        decoy_owner = None

        recovered = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        recovered.close()
    finally:
        monkeypatch.setattr(artifacts.os, "scandir", original_scandir)
        if writer is not None:
            writer.__exit__(RuntimeError, None, None)
        stale_owner.close()
        if decoy_owner is not None:
            decoy_owner.close()


@pytest.mark.parametrize("populated", [False, True])
def test_scanner_position_binding_supports_fresh_directories(
    tmp_path,
    populated,
):
    root = tmp_path / "cache"
    if populated:
        root.mkdir()
        (root / "existing").write_text("value", encoding="utf-8")

    writer = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)

    assert (root / ".artifact_directory_anchor").exists()
    assert writer._lock_fd is not None
    os.fstat(writer._lock_fd)
    writer.close()


@pytest.mark.parametrize("anchor_kind", ["file", "directory"])
def test_existing_anchor_entry_is_accepted(tmp_path, anchor_kind):
    root = tmp_path / "cache"
    root.mkdir()
    anchor = root / ".artifact_directory_anchor"
    if anchor_kind == "file":
        anchor.write_text("preexisting", encoding="utf-8")
    else:
        anchor.mkdir()

    writer = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
    writer.close()


def test_concurrent_anchor_creation_is_accepted(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    original_mkdir = artifacts.os.mkdir
    created_concurrently = False

    def create_then_report_exists(path, mode=0o777, *, dir_fd=None):
        nonlocal created_concurrently
        if Path(path).name == ".artifact_directory_anchor":
            assert not created_concurrently
            created_concurrently = True
            original_mkdir(path, mode, dir_fd=dir_fd)
            raise FileExistsError(artifacts.errno.EEXIST, "already exists", path)
        return original_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(artifacts.os, "mkdir", create_then_report_exists)
    writer = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)

    assert created_concurrently
    writer.close()


def test_empty_path_replacement_between_anchor_and_scanner_retries(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    moved = tmp_path / "moved"
    original_scandir = artifacts.os.scandir
    replaced = False

    def replace_with_empty_directory(path):
        nonlocal replaced
        if Path(path) == root and not replaced:
            replaced = True
            root.rename(moved)
            root.mkdir()
        return original_scandir(path)

    monkeypatch.setattr(
        artifacts.os,
        "scandir",
        replace_with_empty_directory,
    )
    writer = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
    writer.add(_record("a"))
    writer.close()

    assert replaced
    assert (root / "manifest.json").exists()
    assert not (moved / "manifest.json").exists()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_before_candidate_handle_assignment_performs_no_child_io(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    original_scandir = artifacts.os.scandir
    source_lines, first_line = inspect.getsourcelines(
        artifacts._CandidateLease.open
    )
    assignment_line = first_line + next(
        offset
        for offset, line in enumerate(source_lines)
        if line.strip() == "self._directory_handle = _DirectoryHandle(directory)"
    )
    forked = False
    in_child = False
    child_process_id = None
    child_scanned = False

    def record_child_scan(path):
        nonlocal child_scanned
        if in_child and Path(path) == root:
            child_scanned = True
        return original_scandir(path)

    def fork_before_assignment(frame, event, _argument):
        nonlocal child_process_id, forked, in_child
        if (
            not forked
            and frame.f_code is artifacts._CandidateLease.open.__code__
            and event == "line"
            and frame.f_lineno == assignment_line
        ):
            forked = True
            process_id = os.fork()
            if process_id == 0:
                in_child = True
                child_process_id = 0
            else:
                child_process_id = process_id
        return fork_before_assignment

    monkeypatch.setattr(artifacts.os, "scandir", record_child_scan)
    try:
        sys.settrace(fork_before_assignment)
        try:
            writer = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        except RuntimeError as error:
            if not in_child:
                raise
            sys.settrace(None)
            clean_failure = "fork" in str(error)
            os._exit(0 if clean_failure and not child_scanned else 61)

        sys.settrace(None)
        if in_child:
            os._exit(62)
        assert child_process_id is not None
        _waited_id, status = os.waitpid(child_process_id, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        writer.close()
    finally:
        sys.settrace(None)
        if in_child:
            os._exit(63)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_after_position_binding_does_not_mutate_a_reused_fd(tmp_path):
    root = tmp_path / "cache"
    source_lines, first_line = inspect.getsourcelines(
        artifacts._DirectoryHandle.open
    )
    candidate_line = first_line + next(
        offset
        for offset, line in enumerate(source_lines)
        if line.strip() == "candidates = ["
    )
    forked = False
    in_child = False
    child_process_id = None
    reused_descriptor = None
    child_exit_code = None
    original_position = None

    def fork_before_candidate_assignment(frame, event, _argument):
        nonlocal child_exit_code, child_process_id, forked, in_child
        nonlocal original_position, reused_descriptor
        if (
            not forked
            and frame.f_code is artifacts._DirectoryHandle.open.__code__
            and event == "line"
            and frame.f_lineno == candidate_line
        ):
            before = frame.f_locals["before"]
            after = frame.f_locals["after"]
            changed = [
                descriptor
                for descriptor, position in before.items()
                if descriptor in after and after[descriptor] != position
            ]
            assert len(changed) == 1
            forked = True
            process_id = os.fork()
            if process_id == 0:
                in_child = True
                child_process_id = 0
                flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                while reused_descriptor != changed[0]:
                    reused_descriptor = os.open(root, flags)
                    if reused_descriptor > changed[0]:
                        os._exit(70)
                original_position = os.lseek(
                    reused_descriptor,
                    0,
                    os.SEEK_CUR,
                )
            else:
                child_process_id = process_id
                _waited_id, status = os.waitpid(process_id, 0)
                child_exit_code = os.waitstatus_to_exitcode(status)
        return fork_before_candidate_assignment

    try:
        sys.settrace(fork_before_candidate_assignment)
        try:
            writer = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        except BaseException as error:
            if not in_child:
                raise
            sys.settrace(None)
            if not isinstance(error, RuntimeError):
                os._exit(74)
            alive = True
            try:
                os.fstat(reused_descriptor)
            except OSError:
                alive = False
            unchanged = (
                alive
                and os.lseek(reused_descriptor, 0, os.SEEK_CUR)
                == original_position
            )
            probe = os.open(
                root,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            unlocked = True
            try:
                artifacts.fcntl.flock(
                    probe,
                    artifacts.fcntl.LOCK_EX | artifacts.fcntl.LOCK_NB,
                )
            except BlockingIOError:
                unlocked = False
            finally:
                os.close(probe)
            clean = (
                artifacts._PENDING_LOCK_FDS == set()
                and artifacts._PENDING_LOCK_LEASES == set()
            )
            if "fork" not in str(error):
                os._exit(75)
            if not alive:
                os._exit(76)
            if not unchanged:
                os._exit(77)
            if not unlocked:
                os._exit(78)
            if not clean:
                os._exit(79)
            os._exit(0)

        sys.settrace(None)
        if in_child:
            os._exit(72)
        assert child_process_id is not None
        assert child_exit_code == 0
        writer.close()
    finally:
        sys.settrace(None)
        if in_child:
            os._exit(73)


def test_constructor_helper_return_is_exception_safe(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    original_scandir = artifacts.os.scandir
    opened_descriptor = None
    interrupted = False
    target_line = next(
        instruction.positions.lineno
        for instruction in dis.get_instructions(ShardWriter.__init__)
        if instruction.opname == "LOAD_GLOBAL"
        and instruction.argval == "_acquire_directory_lock"
    )

    def capture_open(path):
        nonlocal opened_descriptor
        before = artifacts._live_fd_snapshot()
        owner = original_scandir(path)
        if Path(path) == root:
            candidates = artifacts._live_fd_snapshot() - before
            assert len(candidates) == 1
            opened_descriptor = candidates.pop()
        return owner

    def interrupt_helper_return(frame, event, _argument):
        nonlocal interrupted
        if frame.f_code is ShardWriter.__init__.__code__:
            if (
                not interrupted
                and event == "line"
                and frame.f_lineno == target_line
            ):
                interrupted = True
                raise KeyboardInterrupt("interrupted after directory helper")
        return interrupt_helper_return

    monkeypatch.setattr(artifacts.os, "scandir", capture_open)
    try:
        sys.settrace(interrupt_helper_return)
        with pytest.raises(KeyboardInterrupt, match="after directory helper"):
            ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        sys.settrace(None)

        assert interrupted
        assert opened_descriptor is not None
        with pytest.raises(OSError) as error:
            os.fstat(opened_descriptor)
        assert error.value.errno == artifacts.errno.EBADF
        assert artifacts._PENDING_LOCK_FDS == set()
        assert artifacts._PENDING_LOCK_LEASES == set()
        recovered = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        recovered.close()
    finally:
        sys.settrace(None)
        if opened_descriptor is not None:
            try:
                os.close(opened_descriptor)
            except OSError:
                pass


def test_acquisition_failure_never_recloses_a_reused_candidate_fd(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    original_flock = artifacts.fcntl.flock
    original_close = artifacts.os.close
    original_handle_close = artifacts._DirectoryHandle.close
    candidate_descriptor = None
    replacement_descriptor = None
    close_calls = 0

    def reject_lock(descriptor, operation):
        nonlocal candidate_descriptor
        if operation & artifacts.fcntl.LOCK_EX:
            candidate_descriptor = descriptor
            raise BlockingIOError(artifacts.errno.EAGAIN, "forced contention")
        return original_flock(descriptor, operation)

    def close_then_reuse(handle):
        nonlocal close_calls, replacement_descriptor
        descriptor = handle._descriptor
        if descriptor != candidate_descriptor or handle._owner is None:
            return original_handle_close(handle)
        close_calls += 1
        if close_calls == 1:
            original_handle_close(handle)

            def reuse_in_thread():
                nonlocal replacement_descriptor
                replacement_descriptor = os.open(os.devnull, os.O_RDONLY)

            thread = threading.Thread(target=reuse_in_thread)
            thread.start()
            thread.join()
            assert replacement_descriptor == descriptor
            return
        return original_handle_close(handle)

    monkeypatch.setattr(artifacts.fcntl, "flock", reject_lock)
    monkeypatch.setattr(artifacts._DirectoryHandle, "close", close_then_reuse)
    try:
        with pytest.raises(RuntimeError, match="active writer"):
            ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        assert close_calls == 1
        assert replacement_descriptor is not None
        os.fstat(replacement_descriptor)
    finally:
        monkeypatch.setattr(artifacts.fcntl, "flock", original_flock)
        monkeypatch.setattr(
            artifacts._DirectoryHandle,
            "close",
            original_handle_close,
        )
        if replacement_descriptor is not None:
            try:
                original_close(replacement_descriptor)
            except OSError:
                pass


def test_candidate_initialization_interruption_never_leaks_raw_descriptor(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    original_scandir = artifacts.os.scandir
    opened_descriptor = None
    interrupted = False
    source_lines, first_line = inspect.getsourcelines(
        artifacts._CandidateLease._adopt_open_descriptor
    )
    target_line = first_line + next(
        offset
        for offset, line in enumerate(source_lines)
        if line.strip() == "_PENDING_LOCK_FDS.add(descriptor)"
    )

    def capture_open(path):
        nonlocal opened_descriptor
        before = artifacts._live_fd_snapshot()
        owner = original_scandir(path)
        if Path(path) == root:
            candidates = artifacts._live_fd_snapshot() - before
            assert len(candidates) == 1
            opened_descriptor = candidates.pop()
        return owner

    def interrupt_during_initialization(frame, event, _argument):
        nonlocal interrupted
        if (
            not interrupted
            and frame.f_code
            is artifacts._CandidateLease._adopt_open_descriptor.__code__
            and event == "line"
            and frame.f_lineno == target_line
        ):
            interrupted = True
            raise KeyboardInterrupt("interrupted during candidate initialization")
        return interrupt_during_initialization

    monkeypatch.setattr(artifacts.os, "scandir", capture_open)
    try:
        sys.settrace(interrupt_during_initialization)
        with pytest.raises(KeyboardInterrupt, match="candidate initialization"):
            ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        sys.settrace(None)

        assert interrupted
        assert opened_descriptor is not None
        with pytest.raises(OSError) as error:
            os.fstat(opened_descriptor)
        assert error.value.errno == artifacts.errno.EBADF
        assert artifacts._PENDING_LOCK_FDS == set()
        recovered = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        recovered.close()
    finally:
        sys.settrace(None)
        if opened_descriptor is not None:
            try:
                os.close(opened_descriptor)
            except OSError:
                pass


def test_candidate_open_adoption_recovers_from_every_line_boundary(
    tmp_path,
    monkeypatch,
):
    probe_root = tmp_path / "candidate-adoption-probe"
    probe_root.mkdir()
    probe = artifacts._CandidateLease()
    executed_lines = []
    adoption_codes = {
        artifacts._CandidateLease.open.__code__,
        artifacts._CandidateLease._track_open_descriptor.__code__,
        artifacts._CandidateLease._prepare_open_descriptor.__code__,
        artifacts._CandidateLease._adopt_open_descriptor.__code__,
    }

    def record_adoption_lines(frame, event, _argument):
        if frame.f_code in adoption_codes and event == "line":
            executed_lines.append((frame.f_code, frame.f_lineno))
        return record_adoption_lines

    sys.settrace(record_adoption_lines)
    try:
        probe.open(probe_root)
    finally:
        sys.settrace(None)
        probe.close()
    boundaries = list(dict.fromkeys(executed_lines))
    assert boundaries

    original_scandir = artifacts.os.scandir
    original_close = artifacts.os.close
    for index, (target_code, target_line) in enumerate(boundaries):
        root = tmp_path / f"candidate-adoption-boundary-{index}"
        root.mkdir()
        lease = artifacts._CandidateLease()
        opened_descriptor = None
        interrupted = False

        def capture_open(path):
            nonlocal opened_descriptor
            before = artifacts._live_fd_snapshot()
            owner = original_scandir(path)
            if Path(path) == root:
                candidates = artifacts._live_fd_snapshot() - before
                assert len(candidates) == 1
                opened_descriptor = candidates.pop()
            return owner

        def interrupt_at_boundary(frame, event, _argument):
            nonlocal interrupted
            if (
                not interrupted
                and frame.f_code is target_code
                and event == "line"
                and frame.f_lineno == target_line
            ):
                interrupted = True
                raise KeyboardInterrupt(f"interrupted at line {target_line}")
            return interrupt_at_boundary

        monkeypatch.setattr(artifacts.os, "scandir", capture_open)
        try:
            sys.settrace(interrupt_at_boundary)
            with pytest.raises(KeyboardInterrupt, match="interrupted at line"):
                lease.open(root)
            sys.settrace(None)
            monkeypatch.setattr(artifacts.os, "scandir", original_scandir)

            assert interrupted
            assert artifacts._PENDING_LOCK_FDS == set(), (
                target_code.co_name,
                target_line,
            )
            assert artifacts._PENDING_LOCK_LEASES == set()
            if opened_descriptor is not None:
                try:
                    os.fstat(opened_descriptor)
                except OSError as error:
                    assert error.errno == artifacts.errno.EBADF
                else:
                    pytest.fail((target_code.co_name, target_line))
            recovered = ShardWriter(
                root,
                {"stage": "evaluation"},
                shard_size=1,
            )
            recovered.close()
        finally:
            sys.settrace(None)
            monkeypatch.setattr(artifacts.os, "scandir", original_scandir)
            lease._descriptor = None
            artifacts._PENDING_LOCK_LEASES.discard(lease)
            if opened_descriptor is not None:
                artifacts._PENDING_LOCK_FDS.discard(opened_descriptor)
                try:
                    original_close(opened_descriptor)
                except OSError:
                    pass


def test_open_cleanup_interruption_never_loses_unadopted_descriptor(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    original_scandir = artifacts.os.scandir
    original_prepare = artifacts._CandidateLease._prepare_open_descriptor
    opened_descriptor = None
    interrupted = False
    source_lines, first_line = inspect.getsourcelines(
        artifacts._CandidateLease.open
    )
    cleanup_line = first_line + max(
        offset
        for offset, line in enumerate(source_lines)
        if line.strip() == "self.close()"
    )

    def capture_open(path):
        nonlocal opened_descriptor
        before = artifacts._live_fd_snapshot()
        owner = original_scandir(path)
        if Path(path) == root:
            candidates = artifacts._live_fd_snapshot() - before
            assert len(candidates) == 1
            opened_descriptor = candidates.pop()
        return owner

    def reject_preparation(_lease, _descriptor):
        raise RuntimeError("forced preparation failure")

    def interrupt_cleanup(frame, event, _argument):
        nonlocal interrupted
        if (
            not interrupted
            and frame.f_code is artifacts._CandidateLease.open.__code__
            and event == "line"
            and frame.f_lineno == cleanup_line
        ):
            interrupted = True
            raise KeyboardInterrupt("interrupted during open cleanup")
        return interrupt_cleanup

    monkeypatch.setattr(artifacts.os, "scandir", capture_open)
    monkeypatch.setattr(
        artifacts._CandidateLease,
        "_prepare_open_descriptor",
        reject_preparation,
    )
    try:
        sys.settrace(interrupt_cleanup)
        with pytest.raises(KeyboardInterrupt, match="open cleanup"):
            ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        sys.settrace(None)
        monkeypatch.setattr(
            artifacts._CandidateLease,
            "_prepare_open_descriptor",
            original_prepare,
        )

        assert interrupted
        assert opened_descriptor is not None
        with pytest.raises(OSError) as error:
            os.fstat(opened_descriptor)
        assert error.value.errno == artifacts.errno.EBADF
        assert artifacts._PENDING_LOCK_FDS == set()
        assert artifacts._PENDING_LOCK_LEASES == set()
        recovered = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        recovered.close()
    finally:
        sys.settrace(None)
        monkeypatch.setattr(
            artifacts._CandidateLease,
            "_prepare_open_descriptor",
            original_prepare,
        )
        if opened_descriptor is not None:
            try:
                os.close(opened_descriptor)
            except OSError:
                pass


def test_raw_candidate_close_interruption_uses_ownership_marker(
    tmp_path,
):
    root = tmp_path / "cache"
    root.mkdir()
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(root, flags)
    lease = artifacts._CandidateLease()
    lease._track_open_descriptor(descriptor)
    source_lines, first_line = inspect.getsourcelines(
        artifacts._CandidateLease._close_once
    )
    close_line = first_line + next(
        offset
        for offset, line in enumerate(source_lines)
        if line.strip() == "os.closerange(descriptor, descriptor + 1)"
    )
    interrupted = False

    def interrupt_before_close(frame, event, _argument):
        nonlocal interrupted
        if (
            not interrupted
            and frame.f_code is artifacts._CandidateLease._close_once.__code__
            and event == "line"
            and frame.f_lineno == close_line
        ):
            interrupted = True
            raise KeyboardInterrupt("interrupted before raw candidate close")
        return interrupt_before_close

    try:
        sys.settrace(interrupt_before_close)
        with pytest.raises(KeyboardInterrupt, match="raw candidate close"):
            lease.close()
        sys.settrace(None)

        assert interrupted
        with pytest.raises(OSError) as error:
            os.fstat(descriptor)
        assert error.value.errno == artifacts.errno.EBADF
        assert artifacts._PENDING_LOCK_FDS == set()
        assert artifacts._PENDING_LOCK_LEASES == set()
        recovered = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        recovered.close()
    finally:
        sys.settrace(None)
        lease._descriptor = None
        lease._opening_descriptor = None
        try:
            os.close(descriptor)
        except OSError:
            pass


def test_candidate_close_recovers_from_every_line_boundary(tmp_path):
    probe_root = tmp_path / "candidate-close-probe"
    probe_root.mkdir()
    probe, _identity = artifacts._open_directory(probe_root)
    probe = artifacts._acquire_directory_lock(probe_root, lease=probe)
    executed_lines = []
    close_codes = {
        artifacts._CandidateLease.close.__code__,
        artifacts._CandidateLease._close_safely.__code__,
        artifacts._CandidateLease._close_once.__code__,
        artifacts._CandidateLease._consume.__code__,
    }

    def record_close_lines(frame, event, _argument):
        if frame.f_code in close_codes and event == "line":
            executed_lines.append((frame.f_code, frame.f_lineno))
        return record_close_lines

    sys.settrace(record_close_lines)
    try:
        probe.close()
    finally:
        sys.settrace(None)
    boundaries = list(dict.fromkeys(executed_lines))
    assert boundaries

    original_flock = artifacts.fcntl.flock
    original_close = artifacts.os.close
    for index, (target_code, target_line) in enumerate(boundaries):
        root = tmp_path / f"candidate-close-boundary-{index}"
        root.mkdir()
        lease, _identity = artifacts._open_directory(root)
        lease = artifacts._acquire_directory_lock(root, lease=lease)
        descriptor = lease.descriptor
        interrupted = False

        def interrupt_at_boundary(frame, event, _argument):
            nonlocal interrupted
            if (
                not interrupted
                and frame.f_code is target_code
                and event == "line"
                and frame.f_lineno == target_line
            ):
                interrupted = True
                raise KeyboardInterrupt(f"interrupted at line {target_line}")
            return interrupt_at_boundary

        try:
            sys.settrace(interrupt_at_boundary)
            with pytest.raises(KeyboardInterrupt, match="interrupted at line"):
                lease.close()
            sys.settrace(None)

            assert interrupted
            assert artifacts._PENDING_LOCK_FDS == set(), (
                target_code.co_name,
                target_line,
            )
            recovered = ShardWriter(
                root,
                {"stage": "evaluation"},
                shard_size=1,
            )
            recovered.close()
        finally:
            sys.settrace(None)
            lease._descriptor = None
            artifacts._PENDING_LOCK_LEASES.discard(lease)
            artifacts._PENDING_LOCK_FDS.discard(descriptor)
            try:
                original_flock(descriptor, artifacts.fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                original_close(descriptor)
            except OSError:
                pass


def test_candidate_close_reconciliation_recovers_from_every_line_boundary(
    tmp_path,
    monkeypatch,
):
    close_codes = {
        artifacts._CandidateLease.close.__code__,
        artifacts._CandidateLease._close_safely.__code__,
        artifacts._CandidateLease._close_once.__code__,
        artifacts._CandidateLease._reconcile_close.__code__,
        artifacts._CandidateLease._consume.__code__,
    }

    def record_lines(target):
        def record(frame, event, _argument):
            if frame.f_code in close_codes and event == "line":
                target.append((frame.f_code, frame.f_lineno))
            return record

        return record

    normal_root = tmp_path / "normal-candidate-reconciliation-probe"
    normal_root.mkdir()
    normal_probe, _identity = artifacts._open_directory(normal_root)
    normal_probe = artifacts._acquire_directory_lock(
        normal_root,
        lease=normal_probe,
    )
    normal_lines = []
    sys.settrace(record_lines(normal_lines))
    try:
        normal_probe.close()
    finally:
        sys.settrace(None)

    original_handle_close = artifacts._DirectoryHandle.close
    exception_root = tmp_path / "exception-candidate-reconciliation-probe"
    exception_root.mkdir()
    exception_probe, _identity = artifacts._open_directory(exception_root)
    exception_probe = artifacts._acquire_directory_lock(
        exception_root,
        lease=exception_probe,
    )
    exception_lines = []
    close_calls = 0

    def interrupt_first_close(handle):
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            raise RuntimeError("enter candidate reconciliation")
        original_handle_close(handle)

    monkeypatch.setattr(
        artifacts._DirectoryHandle,
        "close",
        interrupt_first_close,
    )
    sys.settrace(record_lines(exception_lines))
    try:
        with pytest.raises(RuntimeError, match="enter candidate reconciliation"):
            exception_probe.close()
    finally:
        sys.settrace(None)
        monkeypatch.setattr(
            artifacts._DirectoryHandle,
            "close",
            original_handle_close,
        )

    normal_boundaries = set(normal_lines)
    boundaries = [
        boundary
        for boundary in dict.fromkeys(exception_lines)
        if boundary not in normal_boundaries
    ]
    assert boundaries
    assert any(
        code is artifacts._CandidateLease._reconcile_close.__code__
        for code, _line in boundaries
    )

    original_flock = artifacts.fcntl.flock
    for index, (target_code, target_line) in enumerate(boundaries):
        root = tmp_path / f"candidate-reconciliation-boundary-{index}"
        root.mkdir()
        lease, _identity = artifacts._open_directory(root)
        lease = artifacts._acquire_directory_lock(root, lease=lease)
        descriptor = lease.descriptor
        close_calls = 0
        interrupted = False

        def interrupt_at_boundary(frame, event, _argument):
            nonlocal interrupted
            if (
                not interrupted
                and frame.f_code is target_code
                and event == "line"
                and frame.f_lineno == target_line
            ):
                interrupted = True
                raise KeyboardInterrupt(f"interrupted at line {target_line}")
            return interrupt_at_boundary

        monkeypatch.setattr(
            artifacts._DirectoryHandle,
            "close",
            interrupt_first_close,
        )
        sys.settrace(interrupt_at_boundary)
        try:
            with pytest.raises(KeyboardInterrupt, match="interrupted at line"):
                lease.close()
            sys.settrace(None)
            monkeypatch.setattr(
                artifacts._DirectoryHandle,
                "close",
                original_handle_close,
            )

            assert interrupted
            assert artifacts._PENDING_LOCK_FDS == set(), (
                target_code.co_name,
                target_line,
            )
            recovered = ShardWriter(
                root,
                {"stage": "evaluation"},
                shard_size=1,
            )
            recovered.close()
        finally:
            sys.settrace(None)
            monkeypatch.setattr(
                artifacts._DirectoryHandle,
                "close",
                original_handle_close,
            )
            lease._descriptor = None
            artifacts._PENDING_LOCK_LEASES.discard(lease)
            artifacts._PENDING_LOCK_FDS.discard(descriptor)
            try:
                original_flock(descriptor, artifacts.fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(descriptor)
            except OSError:
                pass


@pytest.mark.parametrize("reuse_same_directory", [False, True])
def test_candidate_close_exception_never_recloses_a_reused_fd(
    tmp_path,
    monkeypatch,
    reuse_same_directory,
):
    root = tmp_path / "cache"
    root.mkdir()
    lease, _identity = artifacts._open_directory(root)
    lease = artifacts._acquire_directory_lock(root, lease=lease)
    candidate_descriptor = lease.descriptor
    original_handle_close = artifacts._DirectoryHandle.close
    replacement_descriptor = None
    close_calls = 0

    def close_reuse_and_interrupt(handle):
        nonlocal close_calls, replacement_descriptor
        if handle._owner is None:
            return original_handle_close(handle)
        close_calls += 1
        assert handle._descriptor == candidate_descriptor
        original_handle_close(handle)
        replacement = root if reuse_same_directory else os.devnull
        flags = os.O_RDONLY
        if reuse_same_directory:
            flags |= getattr(os, "O_DIRECTORY", 0)
        replacement_descriptor = os.open(replacement, flags)
        assert replacement_descriptor == candidate_descriptor
        raise KeyboardInterrupt("candidate close completed")

    monkeypatch.setattr(
        artifacts._DirectoryHandle,
        "close",
        close_reuse_and_interrupt,
    )
    try:
        with pytest.raises(KeyboardInterrupt, match="close completed"):
            lease.close()
        monkeypatch.setattr(
            artifacts._DirectoryHandle,
            "close",
            original_handle_close,
        )

        assert close_calls == 1
        assert artifacts._PENDING_LOCK_FDS == set()
        assert replacement_descriptor is not None
        os.fstat(replacement_descriptor)
        recovered = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        recovered.close()
    finally:
        monkeypatch.setattr(
            artifacts._DirectoryHandle,
            "close",
            original_handle_close,
        )
        if replacement_descriptor is not None:
            try:
                os.close(replacement_descriptor)
            except OSError:
                pass


@pytest.mark.parametrize(
    "target_statement",
    ["writer._lock_fd = descriptor", "self._consume(descriptor)"],
)
def test_lock_result_transfer_is_async_exception_safe(
    tmp_path,
    target_statement,
):
    root = tmp_path / "cache"
    source_lines, first_line = inspect.getsourcelines(
        artifacts._CandidateLease.transfer_to
    )
    ownership_line = first_line + next(
        offset
        for offset, line in enumerate(source_lines)
        if line.strip() == target_statement
    )
    interrupted = False

    def interrupt_before_writer_ownership(frame, event, _argument):
        nonlocal interrupted
        if (
            frame.f_code is artifacts._CandidateLease.transfer_to.__code__
            and event == "line"
            and frame.f_lineno == ownership_line
        ):
            interrupted = True
            raise KeyboardInterrupt("interrupted before writer ownership")
        return interrupt_before_writer_ownership

    original_flock = artifacts.fcntl.flock
    try:
        sys.settrace(interrupt_before_writer_ownership)
        with pytest.raises(KeyboardInterrupt, match="before writer ownership"):
            ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        sys.settrace(None)

        assert interrupted
        assert artifacts._PENDING_LOCK_FDS == set()
        recovered = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        recovered.close()
    finally:
        sys.settrace(None)
        for descriptor in list(artifacts._PENDING_LOCK_FDS):
            original_flock(descriptor, artifacts.fcntl.LOCK_UN)
            os.close(descriptor)
            artifacts._PENDING_LOCK_FDS.discard(descriptor)


@pytest.mark.parametrize(
    "target_statement",
    ["writer._lock_fd_marker = marker", "self._consume(descriptor)"],
)
def test_transfer_interruption_never_recloses_a_reused_candidate_fd(
    tmp_path,
    monkeypatch,
    target_statement,
):
    root = tmp_path / "cache"
    original_open = artifacts._open_directory
    original_handle_close = artifacts._DirectoryHandle.close
    candidate_descriptor = None
    replacement_descriptor = None
    interrupted = False
    source_lines, first_line = inspect.getsourcelines(
        artifacts._CandidateLease.transfer_to
    )
    interruption_line = first_line + next(
        offset
        for offset, line in enumerate(source_lines)
        if line.strip() == target_statement
    )

    def capture_candidate(directory, **kwargs):
        nonlocal candidate_descriptor
        lease, identity = original_open(directory, **kwargs)
        candidate_descriptor = lease.descriptor
        return lease, identity

    def close_then_reuse(handle):
        nonlocal replacement_descriptor
        descriptor = handle._descriptor
        if (
            descriptor == candidate_descriptor
            and handle._owner is not None
            and replacement_descriptor is None
        ):
            original_handle_close(handle)
            replacement_descriptor = os.open(os.devnull, os.O_RDONLY)
            assert replacement_descriptor == descriptor
            return
        return original_handle_close(handle)

    def interrupt_before_consume(frame, event, _argument):
        nonlocal interrupted
        if (
            not interrupted
            and frame.f_code is artifacts._CandidateLease.transfer_to.__code__
            and event == "line"
            and frame.f_lineno == interruption_line
        ):
            interrupted = True
            raise KeyboardInterrupt("interrupted before lease consume")
        return interrupt_before_consume

    monkeypatch.setattr(artifacts, "_open_directory", capture_candidate)
    monkeypatch.setattr(artifacts._DirectoryHandle, "close", close_then_reuse)
    try:
        sys.settrace(interrupt_before_consume)
        with pytest.raises(KeyboardInterrupt, match="before lease consume"):
            ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        sys.settrace(None)
        monkeypatch.setattr(
            artifacts._DirectoryHandle,
            "close",
            original_handle_close,
        )

        assert interrupted
        assert replacement_descriptor is not None
        os.fstat(replacement_descriptor)
    finally:
        sys.settrace(None)
        monkeypatch.setattr(
            artifacts._DirectoryHandle,
            "close",
            original_handle_close,
        )
        if replacement_descriptor is not None:
            try:
                os.close(replacement_descriptor)
            except OSError:
                pass


def test_same_process_handoff_does_not_open_an_unused_candidate(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation"}
    writer = ShardWriter(root, metadata, shard_size=1)
    writer.add(_record("a"))

    def unexpected_open(_directory, **_kwargs):
        raise AssertionError("handoff opened an unused candidate descriptor")

    monkeypatch.setattr(artifacts, "_open_directory", unexpected_open)
    resumed = ShardWriter(root, metadata, shard_size=1)
    resumed.close()


def test_handoff_never_double_closes_a_reused_candidate_fd(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation"}
    writer = ShardWriter(root, metadata, shard_size=1)
    writer.add(_record("a"))
    original_open = artifacts._open_directory
    original_close = artifacts.os.close
    candidate_fds = []
    replacement_fd = None

    def capture_candidate(directory, **kwargs):
        descriptor, identity = original_open(directory, **kwargs)
        candidate_fds.append(descriptor)
        return descriptor, identity

    def close_reuse_and_interrupt(descriptor):
        nonlocal replacement_fd
        if (
            replacement_fd is None
            and candidate_fds
            and descriptor == candidate_fds[0]
        ):
            original_close(descriptor)
            replacement_fd = os.open(os.devnull, os.O_RDONLY)
            assert replacement_fd == descriptor
            raise KeyboardInterrupt("close completed before interruption")
        original_close(descriptor)

    monkeypatch.setattr(artifacts, "_open_directory", capture_candidate)
    monkeypatch.setattr(artifacts.os, "close", close_reuse_and_interrupt)
    resumed = None
    try:
        try:
            resumed = ShardWriter(root, metadata, shard_size=1)
        except KeyboardInterrupt as error:
            assert str(error) == "close completed before interruption"
        monkeypatch.setattr(artifacts.os, "close", original_close)

        assert artifacts._PENDING_LOCK_FDS == set()
        if replacement_fd is None:
            assert resumed is not None
        else:
            os.fstat(replacement_fd)
    finally:
        monkeypatch.setattr(artifacts.os, "close", original_close)
        if resumed is not None:
            resumed.close()
        if replacement_fd is not None:
            try:
                original_close(replacement_fd)
            except OSError:
                pass


def test_lock_acquisition_base_exception_releases_pending_fd(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    original_flock = artifacts.fcntl.flock

    def lock_then_interrupt(descriptor, operation):
        original_flock(descriptor, operation)
        if operation & artifacts.fcntl.LOCK_EX:
            raise KeyboardInterrupt("interrupted after lock acquisition")

    monkeypatch.setattr(artifacts.fcntl, "flock", lock_then_interrupt)
    try:
        with pytest.raises(KeyboardInterrupt, match="after lock acquisition"):
            ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        assert artifacts._PENDING_LOCK_FDS == set()
        monkeypatch.setattr(artifacts.fcntl, "flock", original_flock)
        recovered = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        recovered.close()
    finally:
        monkeypatch.setattr(artifacts.fcntl, "flock", original_flock)
        for descriptor in list(artifacts._PENDING_LOCK_FDS):
            original_flock(descriptor, artifacts.fcntl.LOCK_UN)
            os.close(descriptor)
            artifacts._PENDING_LOCK_FDS.discard(descriptor)


def test_release_lock_recovers_from_every_line_boundary(tmp_path):
    probe = ShardWriter(
        tmp_path / "line-probe",
        {"stage": "evaluation"},
        shard_size=1,
    )
    executed_lines = []
    release_codes = {
        ShardWriter._release_lock.__code__,
        ShardWriter._release_lock_safely.__code__,
        ShardWriter._release_lock_once.__code__,
    }

    def record_release_lines(frame, event, _argument):
        if frame.f_code in release_codes and event == "line":
            executed_lines.append((frame.f_code, frame.f_lineno))
        return record_release_lines

    sys.settrace(record_release_lines)
    try:
        probe.__exit__(RuntimeError, None, None)
    finally:
        sys.settrace(None)
    boundaries = list(dict.fromkeys(executed_lines))
    assert boundaries

    original_flock = artifacts.fcntl.flock
    original_close = artifacts.os.close
    for index, (target_code, target_line) in enumerate(boundaries):
        root = tmp_path / f"boundary-{index}"
        writer = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        writer.add(_record("a"))
        descriptor = writer._lock_fd
        interrupted = False

        def interrupt_at_boundary(frame, event, _argument):
            nonlocal interrupted
            if (
                not interrupted
                and frame.f_code is target_code
                and event == "line"
                and frame.f_lineno == target_line
            ):
                interrupted = True
                raise KeyboardInterrupt(f"interrupted at line {target_line}")
            return interrupt_at_boundary

        try:
            sys.settrace(interrupt_at_boundary)
            with pytest.raises(KeyboardInterrupt, match="interrupted at line"):
                writer.__exit__(RuntimeError, None, None)
            sys.settrace(None)

            assert interrupted
            recovered = ShardWriter(
                root,
                {"stage": "evaluation"},
                shard_size=1,
            )
            assert recovered.existing_keys() == {("a", 0)}
            recovered.close()
        finally:
            sys.settrace(None)
            writer._lock_fd = None
            if descriptor is not None:
                try:
                    original_flock(descriptor, artifacts.fcntl.LOCK_UN)
                except OSError:
                    pass
                try:
                    original_close(descriptor)
                except OSError:
                    pass


def test_release_reconciliation_recovers_from_every_line_boundary(
    tmp_path,
    monkeypatch,
):
    release_codes = {
        ShardWriter._release_lock.__code__,
        ShardWriter._release_lock_safely.__code__,
        ShardWriter._release_lock_once.__code__,
        ShardWriter._reconcile_lock_release.__code__,
    }

    def record_lines(target):
        def record(frame, event, _argument):
            if frame.f_code in release_codes and event == "line":
                target.append((frame.f_code, frame.f_lineno))
            return record

        return record

    normal_lines = []
    normal_probe = ShardWriter(
        tmp_path / "normal-reconciliation-probe",
        {"stage": "evaluation"},
        shard_size=1,
    )
    sys.settrace(record_lines(normal_lines))
    try:
        normal_probe.__exit__(RuntimeError, None, None)
    finally:
        sys.settrace(None)

    original_handle_close = artifacts._DirectoryHandle.close
    exception_lines = []
    close_calls = 0

    def interrupt_first_close(handle):
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            raise RuntimeError("enter release reconciliation")
        original_handle_close(handle)

    exception_probe = ShardWriter(
        tmp_path / "exception-reconciliation-probe",
        {"stage": "evaluation"},
        shard_size=1,
    )
    monkeypatch.setattr(artifacts._DirectoryHandle, "close", interrupt_first_close)
    sys.settrace(record_lines(exception_lines))
    try:
        with pytest.raises(RuntimeError, match="enter release reconciliation"):
            exception_probe.__exit__(RuntimeError, None, None)
    finally:
        sys.settrace(None)
        monkeypatch.setattr(artifacts._DirectoryHandle, "close", original_handle_close)

    normal_boundaries = set(normal_lines)
    boundaries = [
        boundary
        for boundary in dict.fromkeys(exception_lines)
        if boundary not in normal_boundaries
    ]
    assert boundaries
    assert any(
        code is ShardWriter._reconcile_lock_release.__code__
        for code, _line in boundaries
    )

    for index, (target_code, target_line) in enumerate(boundaries):
        root = tmp_path / f"reconciliation-boundary-{index}"
        writer = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        writer.add(_record("a"))
        descriptor = writer._lock_fd
        close_calls = 0
        interrupted = False

        def interrupt_at_boundary(frame, event, _argument):
            nonlocal interrupted
            if (
                not interrupted
                and frame.f_code is target_code
                and event == "line"
                and frame.f_lineno == target_line
            ):
                interrupted = True
                raise KeyboardInterrupt(f"interrupted at line {target_line}")
            return interrupt_at_boundary

        monkeypatch.setattr(artifacts._DirectoryHandle, "close", interrupt_first_close)
        sys.settrace(interrupt_at_boundary)
        try:
            with pytest.raises(KeyboardInterrupt, match="interrupted at line"):
                writer.__exit__(RuntimeError, None, None)
            sys.settrace(None)
            monkeypatch.setattr(artifacts._DirectoryHandle, "close", original_handle_close)

            assert interrupted
            recovered = ShardWriter(
                root,
                {"stage": "evaluation"},
                shard_size=1,
            )
            assert recovered.existing_keys() == {("a", 0)}
            recovered.close()
        finally:
            sys.settrace(None)
            monkeypatch.setattr(artifacts._DirectoryHandle, "close", original_handle_close)
            writer._lock_fd = None
            writer._lock_handle = None
            if descriptor is not None:
                os.closerange(descriptor, descriptor + 1)


@pytest.mark.parametrize("reuse_same_directory", [False, True])
def test_release_close_exception_never_recloses_a_reused_fd(
    tmp_path,
    monkeypatch,
    reuse_same_directory,
):
    root = tmp_path / "cache"
    writer = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
    writer.add(_record("a"))
    original_handle_close = artifacts._DirectoryHandle.close
    replacement_fd = None
    close_calls = 0

    def close_reuse_and_interrupt(handle):
        nonlocal close_calls, replacement_fd
        if handle._owner is None:
            return original_handle_close(handle)
        close_calls += 1
        first = handle._descriptor
        assert first is not None
        original_handle_close(handle)
        replacement = root if reuse_same_directory else os.devnull
        flags = os.O_RDONLY
        if reuse_same_directory:
            flags |= getattr(os, "O_DIRECTORY", 0)
        replacement_fd = os.open(replacement, flags)
        assert replacement_fd == first
        raise KeyboardInterrupt("close completed before interruption")

    monkeypatch.setattr(artifacts._DirectoryHandle, "close", close_reuse_and_interrupt)
    try:
        with pytest.raises(KeyboardInterrupt, match="close completed"):
            writer.__exit__(RuntimeError, None, None)
        monkeypatch.setattr(artifacts._DirectoryHandle, "close", original_handle_close)

        assert close_calls == 1
        assert replacement_fd is not None
        os.fstat(replacement_fd)
        recovered = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        assert recovered.existing_keys() == {("a", 0)}
        recovered.close()
    finally:
        monkeypatch.setattr(artifacts._DirectoryHandle, "close", original_handle_close)
        if replacement_fd is not None:
            try:
                os.close(replacement_fd)
            except OSError:
                pass


def test_lock_descriptor_marker_is_nonzero_and_stable(tmp_path):
    writer = ShardWriter(
        tmp_path / "cache",
        {"stage": "evaluation"},
        shard_size=1,
    )
    descriptor = writer._lock_fd
    marker = writer._lock_fd_marker

    assert descriptor is not None
    assert marker is not None and marker > 0
    assert os.lseek(descriptor, 0, os.SEEK_CUR) == marker
    writer.add(_record("a"))
    assert os.lseek(descriptor, 0, os.SEEK_CUR) == marker
    writer.close()


def test_unsupported_lock_descriptor_marker_fails_cleanly(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    original_lseek = artifacts.os.lseek

    def reject_marker(descriptor, offset, whence):
        if whence == os.SEEK_SET and offset != 0:
            raise OSError(artifacts.errno.EINVAL, "marker unsupported")
        return original_lseek(descriptor, offset, whence)

    monkeypatch.setattr(artifacts.os, "lseek", reject_marker)
    with pytest.raises(RuntimeError, match="does not support ownership markers"):
        ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
    monkeypatch.setattr(artifacts.os, "lseek", original_lseek)

    recovered = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
    recovered.close()


def test_release_retries_when_close_did_not_start(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    writer = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
    writer.add(_record("a"))
    descriptor = writer._lock_fd
    original_handle_close = artifacts._DirectoryHandle.close
    close_calls = 0

    def interrupt_before_close(handle):
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            raise KeyboardInterrupt("interrupted before close")
        original_handle_close(handle)

    monkeypatch.setattr(artifacts._DirectoryHandle, "close", interrupt_before_close)
    try:
        with pytest.raises(KeyboardInterrupt, match="before close"):
            writer.__exit__(RuntimeError, None, None)
        monkeypatch.setattr(artifacts._DirectoryHandle, "close", original_handle_close)

        assert close_calls == 2
        recovered = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        assert recovered.existing_keys() == {("a", 0)}
        recovered.close()
    finally:
        monkeypatch.setattr(artifacts._DirectoryHandle, "close", original_handle_close)
        writer._lock_fd = None
        writer._lock_handle = None
        if descriptor is not None:
            os.closerange(descriptor, descriptor + 1)


def test_interrupted_handle_close_cannot_later_close_a_reused_fd(tmp_path):
    root = tmp_path / "cache"
    writer = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
    writer.add(_record("a"))
    descriptor = writer._lock_fd
    source_lines, first_line = inspect.getsourcelines(
        artifacts._DirectoryHandle.close
    )
    close_line = first_line + next(
        offset
        for offset, line in enumerate(source_lines)
        if line.strip() == "owner.close()"
    )
    interrupted = False
    caught = None
    replacement = None

    def interrupt_before_generator_close(frame, event, _argument):
        nonlocal interrupted
        if (
            not interrupted
            and frame.f_code is artifacts._DirectoryHandle.close.__code__
            and event == "line"
            and frame.f_lineno == close_line
        ):
            interrupted = True
            raise KeyboardInterrupt("interrupted before generator close")
        return interrupt_before_generator_close

    try:
        sys.settrace(interrupt_before_generator_close)
        try:
            writer.__exit__(RuntimeError, None, None)
        except KeyboardInterrupt as error:
            caught = error
        finally:
            sys.settrace(None)

        assert interrupted
        assert caught is not None
        assert descriptor is not None
        with pytest.raises(OSError) as closed:
            os.fstat(descriptor)
        assert closed.value.errno == artifacts.errno.EBADF

        replacement = os.open(os.devnull, os.O_RDONLY)
        assert replacement == descriptor
        caught = None
        gc.collect()
        os.fstat(replacement)
    finally:
        sys.settrace(None)
        if replacement is not None:
            try:
                os.close(replacement)
            except OSError:
                pass


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_during_handoff_closes_the_transferred_fd_in_the_child(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation"}
    writer = ShardWriter(root, metadata, shard_size=1)
    writer.add(_record("a"))
    original_take = ShardWriter._take_lock_from
    child_exit_codes = []

    def fork_after_transfer(successor, predecessor):
        descriptor = original_take(successor, predecessor)
        process_id = os.fork()
        if process_id == 0:
            try:
                os.fstat(descriptor)
            except OSError as error:
                os._exit(0 if error.errno == artifacts.errno.EBADF else 2)
            os._exit(1)
        _waited_id, status = os.waitpid(process_id, 0)
        child_exit_codes.append(os.waitstatus_to_exitcode(status))
        return descriptor

    monkeypatch.setattr(ShardWriter, "_take_lock_from", fork_after_transfer)
    resumed = ShardWriter(root, metadata, shard_size=1)
    resumed.close()

    assert child_exit_codes == [0]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_after_fresh_lock_acquisition_closes_the_fd_in_the_child(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    original_acquire = artifacts._acquire_directory_lock
    child_exit_codes = []

    def fork_after_acquire(directory, **kwargs):
        lease = original_acquire(directory, **kwargs)
        descriptor = lease.descriptor
        process_id = os.fork()
        if process_id == 0:
            try:
                os.fstat(descriptor)
            except OSError as error:
                os._exit(0 if error.errno == artifacts.errno.EBADF else 2)
            os._exit(1)
        _waited_id, status = os.waitpid(process_id, 0)
        child_exit_codes.append(os.waitstatus_to_exitcode(status))
        return lease

    monkeypatch.setattr(
        artifacts,
        "_acquire_directory_lock",
        fork_after_acquire,
    )
    writer = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
    writer.close()

    assert child_exit_codes == [0]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_before_lock_assignment_cannot_close_a_reused_fd(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    metadata = {"stage": "evaluation"}
    original_acquire = artifacts._acquire_directory_lock
    child_process_id = None
    in_child = False
    reused_descriptor = None

    def fork_before_assignment(directory, **kwargs):
        nonlocal child_process_id, in_child, reused_descriptor
        lease = original_acquire(directory, **kwargs)
        descriptor = lease.descriptor
        process_id = os.fork()
        if process_id == 0:
            in_child = True
            child_process_id = 0
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            reused_descriptor = os.open(unrelated, flags)
            if reused_descriptor != descriptor:
                os._exit(20)
            return lease
        child_process_id = process_id
        return lease

    monkeypatch.setattr(
        artifacts,
        "_acquire_directory_lock",
        fork_before_assignment,
    )
    try:
        try:
            writer = ShardWriter(root, metadata, shard_size=1)
        except RuntimeError as error:
            if not in_child:
                raise
            clean_failure = "fork" in str(error)
            descriptor_was_not_reclosed = True
            try:
                os.fstat(reused_descriptor)
            except OSError:
                descriptor_was_not_reclosed = False
            os._exit(0 if clean_failure and descriptor_was_not_reclosed else 21)

        if in_child:
            os._exit(22)

        assert child_process_id is not None
        _waited_id, status = os.waitpid(child_process_id, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        writer.add(_record("parent"))
        writer.close()
        assert [record["image_id"] for record in iter_records(root)] == [
            "parent"
        ]
        assert not any(unrelated.iterdir())
    finally:
        if in_child:
            os._exit(23)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
@pytest.mark.parametrize("reuse_same_directory", [False, True])
def test_fork_after_lock_marker_preserves_an_unrelated_fd(
    tmp_path,
    monkeypatch,
    reuse_same_directory,
):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation"}
    original_mark = artifacts._mark_lock_descriptor
    child_process_id = None
    in_child = False
    reused_descriptor = None

    def mark_then_fork(descriptor, planned_marker=None):
        nonlocal child_process_id, in_child, reused_descriptor
        marker = original_mark(descriptor, planned_marker)
        process_id = os.fork()
        if process_id == 0:
            in_child = True
            child_process_id = 0
            replacement = root if reuse_same_directory else os.devnull
            flags = os.O_RDONLY
            if reuse_same_directory:
                flags |= getattr(os, "O_DIRECTORY", 0)
            replacement_descriptor = os.open(replacement, flags)
            if replacement_descriptor == descriptor:
                os._exit(30)
            reused_descriptor = replacement_descriptor
            return marker
        child_process_id = process_id
        return marker

    monkeypatch.setattr(artifacts, "_mark_lock_descriptor", mark_then_fork)
    try:
        try:
            writer = ShardWriter(root, metadata, shard_size=1)
        except RuntimeError as error:
            if not in_child:
                raise
            clean_failure = "fork" in str(error)
            descriptor_was_not_reclosed = True
            try:
                os.fstat(reused_descriptor)
            except OSError:
                descriptor_was_not_reclosed = False
            os._exit(0 if clean_failure and descriptor_was_not_reclosed else 31)

        if in_child:
            os._exit(32)

        assert child_process_id is not None
        _waited_id, status = os.waitpid(child_process_id, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        writer.add(_record("parent"))
        writer.close()
        assert [record["image_id"] for record in iter_records(root)] == [
            "parent"
        ]
    finally:
        if in_child:
            os._exit(33)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_before_handoff_assignment_cannot_reuse_the_writer_fd(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    metadata = {"stage": "evaluation"}
    predecessor = ShardWriter(root, metadata, shard_size=1)
    predecessor.add(_record("a"))
    child_process_id = None
    in_child = False
    reused_descriptor = None

    class ForkAfterExtend(list):
        def extend(self, values):
            nonlocal child_process_id, in_child, reused_descriptor
            super().extend(values)
            descriptor = predecessor._lock_fd
            assert descriptor is not None
            process_id = os.fork()
            if process_id == 0:
                in_child = True
                child_process_id = 0
                flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                replacement_descriptor = os.open(unrelated, flags)
                if replacement_descriptor != descriptor:
                    os.dup2(replacement_descriptor, descriptor)
                    os.close(replacement_descriptor)
                reused_descriptor = descriptor
                return
            child_process_id = process_id

    monkeypatch.setattr(
        artifacts,
        "_TRANSFER_PARTICIPANTS",
        ForkAfterExtend(),
    )
    try:
        try:
            resumed = ShardWriter(root, metadata, shard_size=1)
        except RuntimeError as error:
            if not in_child:
                raise
            clean_failure = "fork" in str(error)
            descriptor_was_not_reclosed = True
            try:
                os.fstat(reused_descriptor)
            except OSError:
                descriptor_was_not_reclosed = False
            unrelated_is_clean = not any(unrelated.iterdir())
            os._exit(
                0
                if clean_failure
                and descriptor_was_not_reclosed
                and unrelated_is_clean
                else 41
            )

        if in_child:
            os._exit(42)

        assert child_process_id is not None
        _waited_id, status = os.waitpid(child_process_id, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        resumed.add(_record("b"))
        resumed.close()
        assert [record["image_id"] for record in iter_records(root)] == [
            "a",
            "b",
        ]
    finally:
        if in_child:
            os._exit(43)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_before_registration_cannot_reuse_the_writer_fd(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    metadata = {"stage": "evaluation"}
    original_register = artifacts._register_writer
    child_process_id = None
    in_child = False
    reused_descriptor = None

    def fork_before_registration(writer):
        nonlocal child_process_id, in_child, reused_descriptor
        descriptor = writer._lock_fd
        assert descriptor is not None
        process_id = os.fork()
        if process_id == 0:
            in_child = True
            child_process_id = 0
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            reused_descriptor = os.open(unrelated, flags)
            if reused_descriptor != descriptor:
                os._exit(10)
            original_register(writer)
            return
        child_process_id = process_id
        original_register(writer)

    monkeypatch.setattr(artifacts, "_register_writer", fork_before_registration)
    try:
        try:
            writer = ShardWriter(root, metadata, shard_size=1)
        except RuntimeError as error:
            if not in_child:
                raise
            clean_failure = "fork" in str(error)
            unrelated_is_clean = not any(unrelated.iterdir())
            descriptor_was_not_reclosed = True
            try:
                os.fstat(reused_descriptor)
            except OSError:
                descriptor_was_not_reclosed = False
            os._exit(
                0
                if clean_failure
                and unrelated_is_clean
                and descriptor_was_not_reclosed
                else 11
            )

        if in_child:
            writer.add(_record("child"))
            writer.close()
            os._exit(12 if (unrelated / "manifest.json").exists() else 13)

        assert child_process_id is not None
        _waited_id, status = os.waitpid(child_process_id, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        writer.add(_record("parent"))
        writer.close()
        assert [record["image_id"] for record in iter_records(root)] == [
            "parent"
        ]
        assert not any(unrelated.iterdir())
    finally:
        if in_child:
            os._exit(14)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_during_transfer_invalidates_both_child_participants(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation"}
    predecessor = ShardWriter(root, metadata, shard_size=1)
    predecessor.add(_record("a"))
    original_register = artifacts._register_writer
    child_exit_codes = []

    def register_then_fork(successor):
        original_register(successor)
        if successor is predecessor or child_exit_codes:
            return
        process_id = os.fork()
        if process_id == 0:
            predecessor_invalid = (
                predecessor._closed and predecessor._lock_fd is None
            )
            successor_invalid = successor._closed and successor._lock_fd is None
            os._exit(0 if predecessor_invalid and successor_invalid else 1)
        _waited_id, status = os.waitpid(process_id, 0)
        child_exit_codes.append(os.waitstatus_to_exitcode(status))

    monkeypatch.setattr(artifacts, "_register_writer", register_then_fork)
    resumed = ShardWriter(root, metadata, shard_size=1)
    resumed.close()

    assert child_exit_codes == [0]


def test_transfer_registration_failure_restores_the_predecessor(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation"}
    predecessor = ShardWriter(root, metadata, shard_size=1)
    predecessor.add(_record("a"))
    original_register = artifacts._register_writer

    def register_then_fail(successor):
        original_register(successor)
        if successor is not predecessor:
            raise RuntimeError("registration failed")

    monkeypatch.setattr(artifacts, "_register_writer", register_then_fail)
    with pytest.raises(RuntimeError, match="registration failed"):
        ShardWriter(root, metadata, shard_size=1)
    monkeypatch.setattr(artifacts, "_register_writer", original_register)

    predecessor.add(_record("b"))
    predecessor.close()
    assert [record["image_id"] for record in iter_records(root)] == ["a", "b"]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_forked_child_drops_inherited_handoff_ownership(tmp_path):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation"}
    writer = ShardWriter(root, metadata, shard_size=1)
    writer.add(_record("a"))

    process_id = os.fork()
    if process_id == 0:
        try:
            inherited_is_invalid = writer._closed and writer._lock_fd is None
            if not inherited_is_invalid:
                os._exit(1)
            try:
                ShardWriter(root, metadata, shard_size=1)
            except RuntimeError as error:
                if "active writer" in str(error):
                    os._exit(0)
                os._exit(2)
            else:
                os._exit(3)
        except BaseException:
            os._exit(4)

    _waited_id, status = os.waitpid(process_id, 0)
    exit_code = os.waitstatus_to_exitcode(status)
    assert exit_code == 0

    writer.add(_record("b"))
    writer.close()
    assert [record["image_id"] for record in iter_records(root)] == ["a", "b"]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_cross_process_writer_lock_is_exclusive_and_recovers_after_exit(tmp_path):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation"}
    ready_read, ready_write = os.pipe()
    release_read, release_write = os.pipe()
    process_id = os.fork()

    if process_id == 0:
        os.close(ready_read)
        os.close(release_write)
        try:
            child_writer = ShardWriter(root, metadata, shard_size=1)
            child_writer.add(_record("a"))
            os.write(ready_write, b"1")
            os.read(release_read, 1)
            os._exit(0)
        except BaseException:
            os._exit(1)

    os.close(ready_write)
    os.close(release_read)
    assert os.read(ready_read, 1) == b"1"
    with pytest.raises(RuntimeError, match="active writer"):
        ShardWriter(root, metadata, shard_size=1)

    os.write(release_write, b"1")
    os.close(release_write)
    _waited_id, status = os.waitpid(process_id, 0)
    assert os.waitstatus_to_exitcode(status) == 0

    resumed = ShardWriter(root, metadata, shard_size=1)
    assert resumed.existing_keys() == {("a", 0)}
    resumed.close()
    assert [record["image_id"] for record in iter_records(root)] == ["a"]


def test_duplicate_keys_are_rejected_in_memory_and_after_resume(tmp_path):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation"}
    with pytest.raises(RuntimeError, match="interrupted"):
        with ShardWriter(root, metadata, shard_size=2) as writer:
            writer.add({"image_id": "a", "value": torch.tensor([1])})
            with pytest.raises(ValueError, match="duplicate record key.*a.*0"):
                writer.add(_record("a", severity=0))
            writer.add(_record("b"))
            raise RuntimeError("interrupted")

    resumed = ShardWriter(root, metadata, shard_size=2)
    with pytest.raises(ValueError, match="duplicate record key.*b.*0"):
        resumed.add(_record("b"))
    resumed.close()

    assert [record["image_id"] for record in iter_records(root)] == ["a", "b"]


def test_writer_snapshots_nested_metadata_at_construction(tmp_path):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation", "config": {"layers": [1]}}
    writer = ShardWriter(root, metadata, shard_size=1)
    metadata["config"]["layers"].append(2)

    writer.add(_record("a"))
    writer.close()

    assert load_manifest(root)["config"] == {"layers": [1]}


class _UnsafeRecordValue:
    pass


def test_writer_rejects_nested_values_that_safe_torch_load_cannot_read(tmp_path):
    root = tmp_path / "cache"

    with pytest.raises(TypeError, match="safe Torch artifact value"):
        with ShardWriter(root, {"stage": "evaluation"}, shard_size=1) as writer:
            writer.add(
                {
                    "image_id": "a",
                    "nested": [{"unsafe": _UnsafeRecordValue()}],
                }
            )

    assert not (root / "manifest.json").exists()
    assert not list(root.glob("shard_*.pt"))


def test_writer_revalidates_nested_values_immediately_before_publication(tmp_path):
    root = tmp_path / "cache"
    nested = []
    writer = ShardWriter(root, {"stage": "evaluation"}, shard_size=2)
    writer.add(
        {
            "image_id": "a",
            "nested": nested,
        }
    )
    nested.append(_UnsafeRecordValue())

    with pytest.raises(TypeError, match="safe Torch artifact value"):
        writer.close()

    assert not (root / "manifest.json").exists()
    assert not list(root.glob("shard_*.pt"))


def test_resume_rejects_safe_loadable_values_outside_the_record_contract(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    name = "shard_00000.pt"
    shard = root / name
    torch.save(
        [{"image_id": "a", "unsafe": torch.nn.Parameter(torch.ones(1))}],
        shard,
    )
    atomic_json(
        {
            "schema_version": 1,
            "stage": "evaluation",
            "shard_size": 1,
            "record_count": 1,
            "shards": [name],
            "shard_sha256": {name: sha256_file(shard)},
        },
        root / "partial_manifest.json",
    )

    with pytest.raises(TypeError, match="safe Torch artifact value"):
        ShardWriter(root, {"stage": "evaluation"}, shard_size=1)


@pytest.mark.parametrize("shard_size", [0, -1, 1.5, True])
def test_shard_size_must_be_a_positive_integer(tmp_path, shard_size):
    with pytest.raises(ValueError, match="shard_size must be a positive integer"):
        ShardWriter(tmp_path / "cache", {}, shard_size=shard_size)


def test_resume_refuses_metadata_or_shard_size_changes(tmp_path):
    root = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="interrupted"):
        with ShardWriter(
            root,
            {"stage": "reference", "input_id": "abc"},
            shard_size=1,
        ) as writer:
            writer.add(_record("a"))
            raise RuntimeError("interrupted")

    with pytest.raises(ValueError, match="input_id"):
        ShardWriter(root, {"stage": "reference", "input_id": "other"}, shard_size=1)
    with pytest.raises(ValueError, match="unexpected"):
        ShardWriter(
            root,
            {"stage": "reference", "input_id": "abc", "unexpected": True},
            shard_size=1,
        )
    with pytest.raises(ValueError, match="shard_size"):
        ShardWriter(root, {"stage": "reference", "input_id": "abc"}, shard_size=2)


@pytest.mark.parametrize(
    "shards",
    [
        ["shard_00001.pt"],
        ["shard_00000.pt", "shard_00000.pt"],
    ],
)
def test_resume_rejects_noncanonical_or_duplicate_shard_sequences(tmp_path, shards):
    root = tmp_path / "cache"
    root.mkdir()
    torch.save([_record("a")], root / "shard_00000.pt")
    torch.save([_record("b")], root / "shard_00001.pt")
    (root / "partial_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "reference",
                "shard_size": 1,
                "record_count": len(shards),
                "shards": shards,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical shard sequence"):
        ShardWriter(root, {"stage": "reference"}, shard_size=1)


def test_iter_records_rejects_duplicate_shard_names(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    torch.save([_record("a")], root / "shard_00000.pt")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_count": 2,
                "shards": ["shard_00000.pt", "shard_00000.pt"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical shard sequence"):
        list(iter_records(root))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("schema_version", 1.0),
        ("shard_size", True),
        ("shard_size", 1.0),
    ],
)
def test_resume_rejects_noninteger_manifest_control_values(
    tmp_path,
    field,
    value,
):
    root = tmp_path / "cache"
    root.mkdir()
    state = {
        "schema_version": 1,
        "stage": "reference",
        "shard_size": 1,
        "record_count": 0,
        "shards": [],
    }
    state[field] = value
    (root / "partial_manifest.json").write_text(
        json.dumps(state),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=field):
        ShardWriter(root, {"stage": "reference"}, shard_size=1)


def test_completed_writer_is_immutable_and_close_is_idempotent(tmp_path):
    root = tmp_path / "cache"
    writer = ShardWriter(root, {"stage": "reference"}, shard_size=2)
    writer.add(_record("a"))
    writer.close()
    writer.close()

    with pytest.raises(RuntimeError, match="closed"):
        writer.add(_record("b"))
    with pytest.raises(FileExistsError, match="already complete"):
        ShardWriter(root, {"stage": "reference"}, shard_size=2)


def test_artifact_reads_always_use_safe_cpu_torch_load(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="interrupted"):
        with ShardWriter(root, {"stage": "reference"}, shard_size=1) as writer:
            writer.add(_record("a"))
            raise RuntimeError("interrupted")

    calls = []
    original_load = torch.load

    def observed_load(*args, **kwargs):
        calls.append(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(artifacts.torch, "load", observed_load)
    resumed = ShardWriter(root, {"stage": "reference"}, shard_size=1)
    assert resumed.existing_keys() == {("a", 0)}
    resumed.close()
    assert [record["image_id"] for record in iter_records(root)] == ["a"]
    assert calls
    assert all(call == {"map_location": "cpu", "weights_only": True} for call in calls)


@pytest.mark.parametrize("schema_version", [None, 2, True])
def test_iter_records_requires_exact_supported_schema_version(
    tmp_path,
    schema_version,
):
    root = tmp_path / "cache"
    with ShardWriter(root, {"stage": "evaluation"}, shard_size=1) as writer:
        writer.add(_record("a"))
    manifest = load_manifest(root)
    if schema_version is None:
        manifest.pop("schema_version")
    else:
        manifest["schema_version"] = schema_version
    atomic_json(manifest, root / "manifest.json")

    with pytest.raises(ValueError, match="schema_version"):
        list(iter_records(root))


def test_iter_records_requires_one_digest_for_every_shard(tmp_path):
    root = tmp_path / "cache"
    with ShardWriter(root, {"stage": "evaluation"}, shard_size=1) as writer:
        writer.add(_record("a"))
    manifest = load_manifest(root)
    manifest.pop("shard_sha256", None)
    atomic_json(manifest, root / "manifest.json")

    with pytest.raises(ValueError, match="shard_sha256"):
        list(iter_records(root))


def test_iter_records_detects_swapped_shard_contents(tmp_path):
    root = tmp_path / "cache"
    with ShardWriter(root, {"stage": "evaluation"}, shard_size=1) as writer:
        writer.add(_record("a"))
        writer.add(_record("b"))
    first = root / "shard_00000.pt"
    second = root / "shard_00001.pt"
    first_bytes = first.read_bytes()
    first.write_bytes(second.read_bytes())
    second.write_bytes(first_bytes)

    with pytest.raises(ValueError, match="SHA-256"):
        list(iter_records(root))


def test_iter_records_loads_the_same_open_shard_file_that_was_verified(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    with ShardWriter(root, {"stage": "evaluation"}, shard_size=1) as writer:
        writer.add(_record("original"))
    shard = root / "shard_00000.pt"
    replacement = tmp_path / "replacement.pt"
    torch.save([_record("replacement")], replacement)
    original_load = torch.load
    swapped = False

    def replace_path_before_load(source, *args, **kwargs):
        nonlocal swapped
        if not swapped and isinstance(source, Path) and source == shard:
            replacement.replace(shard)
            swapped = True
        return original_load(source, *args, **kwargs)

    monkeypatch.setattr(artifacts.torch, "load", replace_path_before_load)

    assert [record["image_id"] for record in iter_records(root)] == ["original"]


def test_resume_detects_a_tampered_published_shard(tmp_path):
    root = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="interrupted"):
        with ShardWriter(root, {"stage": "evaluation"}, shard_size=1) as writer:
            writer.add(_record("a"))
            raise RuntimeError("interrupted")
    shard = root / "shard_00000.pt"
    shard.write_bytes(shard.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="SHA-256"):
        ShardWriter(root, {"stage": "evaluation"}, shard_size=1)


def test_iter_records_rejects_duplicate_record_keys_in_forged_shards(tmp_path):
    root = tmp_path / "cache"
    _write_manual_artifact(root, [_record("a"), _record("a")])

    with pytest.raises(ValueError, match="duplicate record key"):
        list(iter_records(root))


def test_iter_records_validates_every_record_key(tmp_path):
    root = tmp_path / "cache"
    _write_manual_artifact(root, [{"value": torch.tensor([1])}])

    with pytest.raises(ValueError, match="image_id"):
        list(iter_records(root))


def test_iter_records_rejects_symlinked_shards(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    outside = tmp_path / "outside.pt"
    torch.save([_record("a")], outside)
    shard = root / "shard_00000.pt"
    shard.symlink_to(outside)
    atomic_json(
        {
            "schema_version": 1,
            "record_count": 1,
            "shards": [shard.name],
            "shard_sha256": {shard.name: sha256_file(outside)},
        },
        root / "manifest.json",
    )

    with pytest.raises(ValueError, match="regular file"):
        list(iter_records(root))

_EXECUTED: list[str] = []


def _run_on_unpickle():
    _EXECUTED.append("executed")
    return _record("executed")


class _ArbitraryCodePayload:
    def __reduce__(self):
        return (_run_on_unpickle, ())


def test_iter_records_does_not_execute_pickle_payloads(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    torch.save([_ArbitraryCodePayload()], root / "shard_00000.pt")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_count": 1,
                "shards": ["shard_00000.pt"],
                "shard_sha256": {
                    "shard_00000.pt": sha256_file(root / "shard_00000.pt")
                },
            }
        ),
        encoding="utf-8",
    )
    _EXECUTED.clear()

    with pytest.raises(pickle.UnpicklingError):
        list(iter_records(root))

    assert _EXECUTED == []


def test_regular_file_reader_uses_nonblocking_nofollow_descriptor(tmp_path):
    target = tmp_path / "value.json"
    target.write_text('{"value": 1}\n', encoding="utf-8")
    with artifacts._open_regular_file(
        target, error_message="invalid test file"
    ) as handle:
        assert json.load(handle) == {"value": 1}
        status_flags = artifacts.fcntl.fcntl(handle, artifacts.fcntl.F_GETFL)
        descriptor_flags = artifacts.fcntl.fcntl(
            handle, artifacts.fcntl.F_GETFD
        )

    assert status_flags & os.O_ACCMODE == os.O_RDONLY
    assert status_flags & os.O_NONBLOCK
    assert descriptor_flags & artifacts.fcntl.FD_CLOEXEC


def test_c_owned_reader_interrupt_preserves_identical_signature_decoy(
    tmp_path, monkeypatch
):
    target = tmp_path / "value.bin"
    target.write_bytes(b"value")
    original_open = artifacts.os.open
    original_close = os.close
    resource_fd = None
    flags = (
        os.O_RDONLY
        | os.O_NONBLOCK
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )

    decoy_fd = original_open(target, flags)
    original_fileio = artifacts.io.FileIO

    class ObservedFileIO(original_fileio):
        def __init__(self, *args, **kwargs):
            nonlocal resource_fd
            super().__init__(*args, **kwargs)
            resource_fd = self.fileno()

    monkeypatch.setattr(artifacts.io, "FileIO", ObservedFileIO)
    try:
        _interrupt_at_instruction(
            artifacts._open_regular_file_object,
            lambda: artifacts._open_regular_file_object(target),
            lambda _frame, instruction: instruction.opname == "RETURN_VALUE",
        )
        with pytest.raises(OSError):
            os.fstat(resource_fd)
        os.fstat(decoy_fd)
    finally:
        monkeypatch.undo()
        if decoy_fd is not None:
            original_close(decoy_fd)


@pytest.mark.parametrize("same_inode", [False, True])
def test_c_owned_reader_cleanup_does_not_close_reused_fd(
    tmp_path, monkeypatch, same_inode
):
    target = tmp_path / "value.bin"
    target.write_bytes(b"value")
    original_fileio = artifacts.io.FileIO
    original_open = os.open
    resource_fd = None
    replacement_fd = None

    class ReusingFileIO(original_fileio):
        def __del__(self):
            nonlocal resource_fd, replacement_fd
            if not self.closed:
                resource_fd = self.fileno()
                self.close()
                replacement_fd = original_open(
                    target if same_inode else os.devnull,
                    os.O_RDONLY,
                )
                assert replacement_fd == resource_fd

    monkeypatch.setattr(artifacts.io, "FileIO", ReusingFileIO)
    _interrupt_at_instruction(
        artifacts._open_regular_file_object,
        lambda: artifacts._open_regular_file_object(target),
        lambda _frame, instruction: instruction.opname == "RETURN_VALUE",
    )

    assert resource_fd is not None
    assert replacement_fd is not None
    os.fstat(replacement_fd)
    os.close(replacement_fd)


def test_regular_file_generation_change_after_open_closes_raw_candidate(
    tmp_path, monkeypatch
):
    target = tmp_path / "value.bin"
    target.write_bytes(b"value")
    original_helper = artifacts._open_regular_file_object
    original_generation = artifacts._FORK_GENERATION
    descriptor = None

    def open_across_generation_change(path):
        nonlocal descriptor
        handle = original_helper(path)
        descriptor = handle.fileno()
        artifacts._FORK_GENERATION += 1
        return handle

    monkeypatch.setattr(
        artifacts, "_open_regular_file_object", open_across_generation_change
    )
    try:
        with pytest.raises(RuntimeError, match="crossed a fork"):
            artifacts._open_regular_file(
                target,
                error_message="invalid test file",
            )
        assert descriptor is not None
        with pytest.raises(OSError):
            os.fstat(descriptor)
    finally:
        monkeypatch.undo()
        artifacts._FORK_GENERATION = original_generation


def test_reentrant_regular_file_read_is_rejected_before_raw_open(
    tmp_path, monkeypatch
):
    target = tmp_path / "value.bin"
    target.write_bytes(b"value")
    nested_rejected = False

    def interrupt_with_nested_read(path):
        nonlocal nested_rejected
        try:
            artifacts._open_regular_file(
                target,
                error_message="invalid test file",
            )
        except RuntimeError as error:
            nested_rejected = "reentrant" in str(error)
        raise KeyboardInterrupt("interrupt outer acquisition")

    monkeypatch.setattr(
        artifacts,
        "_open_regular_file_object",
        interrupt_with_nested_read,
    )
    with pytest.raises(KeyboardInterrupt, match="outer acquisition"):
        artifacts._open_regular_file(
            target,
            error_message="invalid test file",
        )

    assert nested_rejected


def test_c_owned_reader_adoption_has_no_python_raw_descriptor_local():
    function = artifacts._open_regular_file_object
    instructions = list(dis.get_instructions(function))

    assert not any(
        instruction.opname.startswith("STORE")
        and instruction.argval in {"descriptor", "fd"}
        for instruction in instructions
    )
    assert "FileIO" in function.__code__.co_names
    assert "_LIBC_OPEN" in function.__code__.co_names


def test_regular_reader_audit_rejection_cannot_leak_descriptor(tmp_path):
    target = tmp_path / "value.bin"
    target.write_bytes(b"value")
    script = """
import gc
import sys

import differential_uncertainty.artifacts as artifacts

target = sys.argv[1]
before = artifacts._live_fd_snapshot()

def reject_integer_fileio(event, arguments):
    if event == "open" and isinstance(arguments[0], int):
        raise RuntimeError("reject integer FileIO adoption")

sys.addaudithook(reject_integer_fileio)
with artifacts._open_regular_file(
    target, error_message="invalid test file"
) as handle:
    assert handle.read() == b"value"
gc.collect()
assert artifacts._live_fd_snapshot() == before
"""

    completed = subprocess.run(
        [sys.executable, "-c", script, os.fspath(target)],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr


def test_regular_reader_does_not_depend_on_numeric_fd_snapshots(
    tmp_path, monkeypatch
):
    target = tmp_path / "value.bin"
    target.write_bytes(b"value")

    def forbid_snapshot():
        raise AssertionError("regular reader took a numeric fd snapshot")

    monkeypatch.setattr(artifacts, "_live_fd_snapshot", forbid_snapshot)
    with artifacts._open_regular_file(
        target, error_message="invalid test file"
    ) as handle:
        assert handle.read() == b"value"


def test_regular_reader_guard_is_not_poisoned_after_store_interrupt():
    function = artifacts._serialized_regular_file_acquisition.__wrapped__
    instructions = list(dis.get_instructions(function))
    active_store = next(
        index
        for index, instruction in enumerate(instructions)
        if instruction.opname == "STORE_ATTR"
        and instruction.argval == "active"
    )
    after_store = instructions[active_store + 1].offset

    def invoke():
        with artifacts._serialized_regular_file_acquisition():
            pass

    _interrupt_at_instruction(
        function,
        invoke,
        lambda _frame, instruction: instruction.offset == after_store,
    )
    invoke()


def test_live_reader_weakrefs_are_callback_free_and_pruned(tmp_path):
    target = tmp_path / "value.bin"
    target.write_bytes(b"value")
    handle = artifacts._open_regular_file(
        target, error_message="invalid test file"
    )
    reference = next(
        reference
        for reference in artifacts._LIVE_READER_LEASES
        if reference() is handle
    )
    assert reference.__callback__ is None

    handle.close()
    handle = None
    gc.collect()

    assert reference() is None
    assert reference in artifacts._LIVE_READER_LEASES
    _invoke_public_reader(target)
    assert reference not in artifacts._LIVE_READER_LEASES


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_regular_reader_trace_interrupt_preserves_baseexception(
    tmp_path, exception_type
):
    target = tmp_path / "value.bin"
    target.write_bytes(b"value")
    decoy_fd = os.open(target, os.O_RDONLY)
    before = artifacts._live_fd_snapshot()
    converter = artifacts._RegularFileOpenFlags.from_param
    traceable_converter = getattr(converter, "__func__", converter)
    converter_code = getattr(traceable_converter, "__code__", None)
    helper = artifacts._open_regular_file_object
    helper_instructions = {
        instruction.offset: instruction
        for instruction in dis.get_instructions(helper)
    }
    interrupted = False

    def interrupt(frame, event, _argument):
        nonlocal interrupted
        frame.f_trace_opcodes = True
        instruction = helper_instructions.get(frame.f_lasti)
        in_python_converter = (
            converter_code is not None and frame.f_code is converter_code
        )
        after_atomic_open = (
            converter_code is None
            and frame.f_code is helper.__code__
            and instruction is not None
            and instruction.opname == "RETURN_VALUE"
        )
        if (
            not interrupted
            and (
                (in_python_converter and event in ("line", "opcode"))
                or (after_atomic_open and event in ("opcode", "return"))
            )
        ):
            interrupted = True
            raise exception_type("injected reader cancellation")
        return interrupt

    observed = None
    try:
        sys.settrace(interrupt)
        with artifacts._open_regular_file(
            target, error_message="invalid test file"
        ) as handle:
            handle.read()
    except BaseException as error:
        observed = error
    finally:
        sys.settrace(None)

    try:
        assert interrupted
        assert type(observed) is exception_type
        assert str(observed) == "injected reader cancellation"
        observed.__traceback__ = None
        gc.collect()
        assert artifacts._live_fd_snapshot() == before
        os.fstat(decoy_fd)
        with artifacts._open_regular_file(
            target, error_message="invalid test file"
        ) as handle:
            assert handle.read() == b"value"
    finally:
        os.close(decoy_fd)


@pytest.mark.parametrize("exception_name", ["KeyboardInterrupt", "SystemExit"])
def test_regular_reader_signal_interrupt_preserves_baseexception(
    tmp_path, exception_name
):
    target = tmp_path / "value.bin"
    target.write_bytes(b"value")
    script = f"""
import os
import signal
import sys

import differential_uncertainty.artifacts as artifacts

target = sys.argv[1]
exception_type = {exception_name}
decoy_fd = os.open(target, os.O_RDONLY)
before = artifacts._live_fd_snapshot()
interruptions = 0
unexpected = None

def interrupt(_signum, _frame):
    raise exception_type("injected reader signal")

signal.signal(signal.SIGALRM, interrupt)
for _ in range(4000):
    handle = None
    try:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0.000005)
            with artifacts._serialized_regular_file_acquisition():
                handle = artifacts._open_regular_file_object(target)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    except (KeyboardInterrupt, SystemExit) as error:
        if (
            type(error) is not exception_type
            or str(error) != "injected reader signal"
        ):
            unexpected = error
            break
        interruptions += 1
    except BaseException as error:
        unexpected = error
        break
    finally:
        if handle is not None:
            handle.close()

assert interruptions
assert unexpected is None, repr(unexpected)
assert artifacts._live_fd_snapshot() == before
os.fstat(decoy_fd)
with artifacts._open_regular_file(
    target, error_message="invalid test file"
) as handle:
    assert handle.read() == b"value"
os.close(decoy_fd)
"""

    completed = subprocess.run(
        [sys.executable, "-c", script, os.fspath(target)],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr


def _invoke_public_reader(target: Path) -> None:
    with artifacts._open_regular_file(
        target, error_message="invalid test file"
    ) as handle:
        assert handle.read() == b"value"


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_public_reader_cancellation_is_never_swallowed(
    tmp_path, exception_type
):
    target = tmp_path / "value.bin"
    target.write_bytes(b"value")
    decoy_fd = os.open(target, os.O_RDONLY)
    before = artifacts._live_fd_snapshot()
    finalizer = artifacts._RegularFileLease.__dict__.get("__del__")
    finalizer_code = None if finalizer is None else finalizer.__code__
    public_code = artifacts._open_regular_file.__code__
    interrupted = False
    handle = None
    observed = None

    def interrupt(frame, event, _argument):
        nonlocal interrupted
        frame.f_trace_opcodes = True
        in_finalizer = (
            finalizer_code is not None
            and frame.f_code is finalizer_code
            and event in ("line", "opcode")
        )
        after_public_return = (
            finalizer_code is None
            and frame.f_code is public_code
            and event == "return"
        )
        if not interrupted and (in_finalizer or after_public_return):
            interrupted = True
            raise exception_type("injected public reader cancellation")
        return interrupt

    try:
        sys.settrace(interrupt)
        handle = artifacts._open_regular_file(
            target, error_message="invalid test file"
        )
    except BaseException as error:
        observed = error
    finally:
        sys.settrace(None)
        if handle is not None:
            handle.close()

    try:
        assert interrupted
        assert type(observed) is exception_type
        assert str(observed) == "injected public reader cancellation"
        observed.__traceback__ = None
        gc.collect()
        assert artifacts._live_fd_snapshot() == before
        os.fstat(decoy_fd)
        _invoke_public_reader(target)
    finally:
        os.close(decoy_fd)


@pytest.mark.parametrize("exception_name", ["KeyboardInterrupt", "SystemExit"])
def test_public_reader_signal_cancellation_is_never_swallowed(
    tmp_path, exception_name
):
    target = tmp_path / "value.bin"
    target.write_bytes(b"value")
    script = f"""
import gc
import os
import signal
import sys

import differential_uncertainty.artifacts as artifacts

target = sys.argv[1]
exception_type = {exception_name}
decoy_fd = os.open(target, os.O_RDONLY)
before = artifacts._live_fd_snapshot()
deliveries = 0
caught = 0
unexpected = None

def interrupt(_signum, _frame):
    global deliveries
    deliveries += 1
    raise exception_type("injected public reader signal")

signal.signal(signal.SIGALRM, interrupt)
for _ in range(4000):
    handle = None
    try:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0.000005)
            handle = artifacts._open_regular_file(
                target, error_message="invalid test file"
            )
            handle.close()
            handle = None
            gc.collect()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    except (KeyboardInterrupt, SystemExit) as error:
        if (
            type(error) is not exception_type
            or str(error) != "injected public reader signal"
        ):
            unexpected = error
            break
        caught += 1
    except BaseException as error:
        unexpected = error
        break
    finally:
        if handle is not None:
            handle.close()

assert deliveries
assert caught == deliveries
assert unexpected is None, repr(unexpected)
assert artifacts._live_fd_snapshot() == before
os.fstat(decoy_fd)
with artifacts._open_regular_file(
    target, error_message="invalid test file"
) as handle:
    assert handle.read() == b"value"
os.close(decoy_fd)
"""

    completed = subprocess.run(
        [sys.executable, "-c", script, os.fspath(target)],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr


def _executed_trace_boundaries(function, invoke) -> list[tuple[str, int]]:
    opcode_offsets = set()
    line_offsets = set()

    def collect(frame, event, _argument):
        if frame.f_code is function.__code__:
            frame.f_trace_opcodes = True
            if event == "opcode":
                opcode_offsets.add(frame.f_lasti)
            elif event == "line":
                line_offsets.add(frame.f_lasti)
        return collect

    try:
        sys.settrace(collect)
        invoke()
    finally:
        sys.settrace(None)
    offsets = opcode_offsets if opcode_offsets else line_offsets
    event = "opcode" if opcode_offsets else "line"
    return [(event, offset) for offset in sorted(offsets)]


@pytest.mark.parametrize("method_name", ["acquire", "release"])
@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_reader_ownership_survives_every_method_boundary_without_finalizer(
    tmp_path, method_name, exception_type
):
    assert "__del__" not in artifacts._RegularFileLease.__dict__
    target = tmp_path / f"{method_name}-{exception_type.__name__}.bin"
    target.write_bytes(b"value")
    method = getattr(artifacts._RegularFileLease, method_name)
    boundaries = _executed_trace_boundaries(
        method, lambda: _invoke_public_reader(target)
    )
    assert boundaries
    decoy_fd = os.open(target, os.O_RDONLY)
    baseline = artifacts._live_fd_snapshot()

    try:
        for target_event, target_offset in boundaries:
            interrupted = False
            observed = None

            def interrupt(frame, event, _argument):
                nonlocal interrupted
                if frame.f_code is method.__code__:
                    frame.f_trace_opcodes = True
                    if (
                        not interrupted
                        and event == target_event
                        and frame.f_lasti == target_offset
                    ):
                        interrupted = True
                        raise exception_type(
                            f"injected {method_name} cancellation"
                        )
                return interrupt

            try:
                sys.settrace(interrupt)
                _invoke_public_reader(target)
            except BaseException as error:
                observed = error
            finally:
                sys.settrace(None)

            assert interrupted, (target_event, target_offset)
            assert type(observed) is exception_type
            observed.__traceback__ = None
            gc.collect()
            assert artifacts._live_fd_snapshot() == baseline
            os.fstat(decoy_fd)
            _invoke_public_reader(target)
    finally:
        os.close(decoy_fd)


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_live_reader_gc_cancellation_propagates_during_pruning(
    tmp_path, exception_type
):
    target = tmp_path / f"gc-{exception_type.__name__}.bin"
    target.write_bytes(b"value")
    decoy_fd = os.open(target, os.O_RDONLY)
    baseline = artifacts._live_fd_snapshot()
    handle = artifacts._open_regular_file(
        target, error_message="invalid test file"
    )
    reference = next(
        reference
        for reference in artifacts._LIVE_READER_LEASES
        if reference() is handle
    )
    callback = reference.__callback__
    prune = getattr(artifacts, "_prune_dead_live_readers", None)
    trace_code = callback.__code__ if callback is not None else prune.__code__
    interrupted = False
    observed = None

    def interrupt(frame, event, _argument):
        nonlocal interrupted
        if frame.f_code is trace_code:
            frame.f_trace_opcodes = True
            if not interrupted and event in ("line", "opcode"):
                interrupted = True
                raise exception_type("injected live-reader GC cancellation")
        return interrupt

    handle.close()
    if callback is None:
        handle = None
        gc.collect()
        assert reference() is None
        assert reference in artifacts._LIVE_READER_LEASES
    try:
        sys.settrace(interrupt)
        if callback is None:
            _invoke_public_reader(target)
        else:
            handle = None
            gc.collect()
    except BaseException as error:
        observed = error
    finally:
        sys.settrace(None)
        handle = None
        gc.collect()

    try:
        assert interrupted
        assert type(observed) is exception_type
        assert str(observed) == "injected live-reader GC cancellation"
        observed.__traceback__ = None
        gc.collect()
        assert artifacts._live_fd_snapshot() == baseline
        os.fstat(decoy_fd)
        _invoke_public_reader(target)
        assert reference not in artifacts._LIVE_READER_LEASES
    finally:
        os.close(decoy_fd)
