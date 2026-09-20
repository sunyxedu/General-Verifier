"""Private Linux temporary-path setup before the native seat sandbox.

Only the host invokes this setup. It grants no proc access or new tool to
the seat, and authenticates no target or result. /tmp names refer to owned
scratch, never the parent's shared temporary directory.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import json
import os
from pathlib import Path
import stat


def _remove_unsafe_entries(directory, prefix="", omissions=None):
    """Do not hand symlinks or special files to the legacy text snapshot."""
    if omissions is None:
        omissions = []
    for name in os.listdir(directory):
        info = os.stat(name, dir_fd=directory, follow_symlinks=False)
        relative = prefix + name
        if stat.S_ISDIR(info.st_mode):
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                            | os.O_CLOEXEC, dir_fd=directory)
            try:
                _remove_unsafe_entries(child, relative + "/", omissions)
            finally:
                os.close(child)
        elif not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            os.unlink(name, dir_fd=directory)
            if len(omissions) < 200:
                omissions.append(relative)
    return omissions


_INNER = r'''
import os,subprocess,sys
fd=int(sys.argv[1])
result=subprocess.run(['/usr/bin/mount','--bind','/proc/self/fd/'+str(fd),'/tmp'],
                      pass_fds=(fd,),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
if result.returncode:
 sys.stderr.write('gv-private-tmp: namespace binding failed\n');sys.exit(125)
os.close(fd)
os.execvp(sys.argv[2],sys.argv[2:])
'''

_OUTER = r'''
import os,pathlib,stat,subprocess,sys
path=sys.argv[1]
flags=os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC
fd=None
try:
 if not path.startswith('/') or str(pathlib.Path(path))!=path or '..' in pathlib.Path(path).parts:
  raise OSError('invalid scratch path')
 fd=os.open('/',flags)
 for part in pathlib.Path(path).parts[1:]:
  child=os.open(part,flags,dir_fd=fd);os.close(fd);fd=child
 info=os.fstat(fd)
 if (not stat.S_ISDIR(info.st_mode) or info.st_uid!=os.geteuid() or info.st_mode&0o077
     or (info.st_dev,info.st_ino)!=(int(sys.argv[2]),int(sys.argv[3]))):
  raise OSError('untrusted scratch')
 result=subprocess.run(['/usr/bin/unshare','--user','--map-root-user','--mount','--fork',
                        sys.executable,'-I','-c',INNER_SOURCE,str(fd),*sys.argv[4:]],
                       pass_fds=(fd,))
 sys.exit(result.returncode if result.returncode>=0 else 128-result.returncode)
except OSError:
 sys.stderr.write('gv-private-tmp: namespace setup failed\n');sys.exit(125)
finally:
 if fd is not None:os.close(fd)
'''


class PrivateTmp:
    """One host-created directory; discharge retains it until its snapshot."""

    def __init__(self, parent, *, retain_parent=None):
        self.path = tempfile.mkdtemp(prefix="gv-private-tmp-", dir=parent)
        info = os.lstat(self.path)
        self.identity = info.st_dev, info.st_ino
        self.retain_fd = None
        try:
            self.capture_path = tempfile.mkdtemp(prefix="gv-cli-capture-", dir=parent)
            if retain_parent is not None:
                self.retain_fd = os.open(retain_parent, os.O_RDONLY | os.O_DIRECTORY
                                         | os.O_NOFOLLOW | os.O_CLOEXEC)
        except OSError:
            shutil.rmtree(self.path)
            if hasattr(self, "capture_path"):
                shutil.rmtree(self.capture_path)
            raise

    def command(self, command):
        source = "INNER_SOURCE=" + repr(_INNER) + "\n" + _OUTER
        return [sys.executable, "-I", "-c", source, self.path,
                *map(str, self.identity), *command]

    def close(self):
        try:
            if self.retain_fd is None:
                shutil.rmtree(self.path)
            else:
                name = Path(self.path).name
                temporary = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY
                                    | os.O_NOFOLLOW | os.O_CLOEXEC)
                try:
                    info = os.fstat(temporary)
                    if (info.st_dev, info.st_ino) != self.identity:
                        raise OSError("private temporary directory changed")
                    omissions = _remove_unsafe_entries(temporary)
                finally:
                    os.close(temporary)
                os.rename(self.path, name, dst_dir_fd=self.retain_fd)
                manifest = {
                    "schema": "gv-private-tmp-retention-v1", "runtime_root": "/tmp",
                    "retained_root": name, "basis": "transport_only",
                    "contents_origin": "process_writable_temporary_files",
                    "complete_capture_implied": False,
                    "omitted_symlink_special_or_shared_entries": omissions,
                }
                fd = os.open(name + ".binding.json", os.O_WRONLY | os.O_CREAT
                             | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                             0o600, dir_fd=self.retain_fd)
                with os.fdopen(fd, "w") as stream:
                    json.dump(manifest, stream, sort_keys=True)
        finally:
            if self.retain_fd is not None:
                os.close(self.retain_fd)
                self.retain_fd = None
            shutil.rmtree(self.capture_path)
