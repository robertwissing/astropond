from types import SimpleNamespace

from sph_interp._arepo import arepo_mesh


def test_darwin_dependency_check_ignores_dyld_cache(monkeypatch, tmp_path):
    exe = tmp_path / "Arepo"
    exe.write_bytes(b"not needed: otool is mocked")
    output = f"""{exe}:
\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0)
\t/opt/homebrew/opt/hdf5/lib/libhdf5.999.dylib (compatibility version 999.0.0)
"""
    monkeypatch.setattr(arepo_mesh.sys, "platform", "darwin")
    monkeypatch.setattr(
        arepo_mesh.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=output))

    assert arepo_mesh._missing_shared_libraries(str(exe)) == [
        "/opt/homebrew/opt/hdf5/lib/libhdf5.999.dylib"]


def test_cubic_binary_with_stale_dependency_is_rebuilt(monkeypatch):
    exe = "/fake/Arepo"
    monkeypatch.setattr(arepo_mesh, "_AREPO_EXE", exe)
    monkeypatch.setattr(arepo_mesh, "has_arepo", lambda: True)
    monkeypatch.setattr(
        arepo_mesh, "_missing_shared_libraries",
        lambda path: ["/opt/homebrew/opt/hdf5/lib/libhdf5.old.dylib"])
    called = {}

    def fake_build(path, config, build_dir, reason):
        called.update(path=path, config=config, build_dir=build_dir,
                      reason=reason)
        return path

    monkeypatch.setattr(arepo_mesh, "_run_arepo_build", fake_build)

    assert arepo_mesh._ensure_arepo_binary(1.0, 1.0) == exe
    assert called["path"] == exe
    assert called["build_dir"] == "build"
    assert "libhdf5.old.dylib" in called["reason"]
