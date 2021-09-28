# Python std.
import re
import os
import random
import string
import logging
import datetime


def ls(path, exts=None, pattern_incl=None, pattern_excl=None,
       ignore_dot_underscore=True):
    """ Lists the directory and returns it sorted. Only the files with
    extensions in `ext` are kept. The output should match the output of Linux
    command "ls". It wrapps os.listdir() which is not guaranteed to produce
    alphanumerically sorted items.

    Args:
        path (str): Absolute or relative path to list.
        exts (str or list of str or None): Extension(s). If None, files with
            any extension are listed. Each e within `exts` can (but does
            not have to) start with a '.' character. E.g. both
            '.tiff' and 'tiff' are allowed.
        pattern_incl (str): regexp pattern, if not found in the file name,
            the file is not listed.
        pattern_excl (str): regexp pattern, if found in the file name,
            the file is not listed.
        ignore_dot_underscore (bool): Whether to ignore files starting with
            '._' (usually spurious files appearing after manipulating the
            linux file system using sshfs)

    Returns:
        list of str: Alphanumerically sorted list of files contained in
        directory `path` and having extension `ext`.
    """
    if isinstance(exts, str):
        exts = [exts]

    files = [f for f in sorted(os.listdir(path))]

    if exts is not None:
        # Include patterns.
        extsstr = ''
        for e in exts:
            extsstr += ('.', '')[e.startswith('.')] + '{}|'.format(e)
        patt_ext = '({})$'.format(extsstr[:-1])
        re_ext = re.compile(patt_ext)
        files = [f for f in files if re_ext.search(f)]

    if ignore_dot_underscore:
        re_du = re.compile('^\._')
        files = [f for f in files if not re_du.match(f)]

    if pattern_incl is not None:
        re_incl = re.compile(pattern_incl)
        files = [f for f in files if re_incl.search(f)]

    if pattern_excl is not None:
        re_excl = re.compile(pattern_excl)
        files = [f for f in files if not re_excl.search(f)]

    return files


def lsd(path, pattern_incl=None, pattern_excl=None):
    """ Lists directories within path.

    Args:
        path (str): Absolue path to containing dir.
        pattern_incl (str): regexp pattern, if not found in the dir name,
            the dir is not listed.
        pattern_excl (str): regexp pattern, if found in the dir name,
            the dir is not listed.

    Returns:
        list: Directories within `path`.
    """
    if pattern_incl is None and pattern_excl is None:
        dirs = [d for d in ls(path) if os.path.isdir(jn(path, d))]
    else:
        if pattern_incl is None:
            pattern_incl = '^.'
        if pattern_excl is None:
            pattern_excl = '^/'

        pincl = re.compile(pattern_incl)
        pexcl = re.compile(pattern_excl)
        dirs = [d for d in ls(path) if
                os.path.isdir(jn(path, d)) and
                pincl.search(d) is not None and
                pexcl.search(d) is None]

    return dirs


def jn(*parts):
    """ Returns the file system path composed of `parts`.

    Args:
        *parts (str): Path parts.

    Returns:
        str: Full path.
    """
    return os.path.join(*parts)


def make_dir(path):
    """ Creates directory `path`. If already exists, does nothing.

    Args:
        path (str): Path to the directory to be created.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def split_name_ext(fname):
    """ Splits the file name to its name and extension.

    Args:
        fname (str): File name without suffix (and without '.').

    Returns:
        str: Name without the extension.
        str: Extension.
    """
    parts = fname.rsplit('.', 1)
    name = parts[0]

    if len(parts) > 1:
        ext = parts[1]
    else:
        ext = ''

    return name, ext


def add_time_suffix(name, keep_extension=True):
    """ Adds the current system time suffix to the file name.
    If `keep_extension`, then the suffix is added before the extension
    (including the ".") if there is any.

    Args:
        name (str): File name.
        keep_extension (bool): Add the suffix before the extension?

    Returns:
        str: New file name.
    """

    # Get time suffix.
    time_suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

    # Generate new name.
    if keep_extension:
        n, e = split_name_ext(name)
        new_name = n + '_' + time_suffix + ('', '.{}'.format(e))[len(e) > 0]
    else:
        new_name = name + '_' + time_suffix

    return new_name


def add_random_suffix(name, length=5, keep_extension=True):
    """ Adds the random string suffix of the form '_****' to a file name,
    where * stands for an upper case ASCII letter or a digit.
    If `keep_extension`, then the suffix is added before the extension
    (including the ".") if there is any.

    Args:
        name (str): File name.
        length (int32): Length of the suffix: 1 letter for underscore,
            the rest for alphanumeric characters.
        keep_extension (bool): Add the suffix before the extension?

    Returns:
        str: New name.
    """
    # Check suffix length.
    if length < 2:
        logging.warning('Suffix must be at least of length 2, '
                        'using "length = 2"')

    # Get random string suffix.
    s = ''.join(random.choice(string.ascii_uppercase + string.digits)
                for _ in range(length - 1))

    # Generate new name.
    if keep_extension:
        n, e = split_name_ext(name)
        new_name = n + '_' + s + ('', '.{}'.format(e))[len(e) > 0]
    else:
        new_name = name + '_' + s

    return new_name


def unique_dir_name(d):
    """ Checks if the `dir` already exists and if so, generates a new name
     by adding current system time as its suffix. If it is still duplicate,
     it adds a random string as a suffix and makes sure it is unique. If
     `dir` is unique already, it is not changed.

    Args:
        d (str): Absolute path to `dir`.

    Returns:
        str: Unique directory name.
    """
    unique_dir = d

    if os.path.exists(d):
        # Add time suffix.
        dir_name = add_time_suffix(d, keep_extension=False)

        # Add random string suffix until the file is unique in the folder.
        unique_dir = dir_name
        while os.path.exists(unique_dir):
            unique_dir += add_random_suffix(unique_dir, keep_extension=False)

    return unique_dir


def unique_file_name(file):
    """ Checks if the `file` already exists and if so, generates a new name
     by adding current system time as its suffix. If it is still duplicate,
     it adds a random string as a suffix and makes sure it is unique. If
     `file` is unique already, it is not changed.

    Args:
        file (str): Absolute path to file.

    Returns:
        str: File name which is unique in its directory.
    """
    unique_file = file

    if os.path.exists(file):
        # Get path and file name.
        path = os.path.dirname(file)
        fname = os.path.basename(file)

        # Add time suffix.
        fname = add_time_suffix(fname)

        # Add random string suffix until the file is unique in the folder.
        new_name = fname
        while os.path.exists(os.path.join(path, new_name)):
            new_name += add_random_suffix(fname)

        # Reconstruct the whole path.
        unique_file = os.path.join(path, new_name)
    return unique_file


def split_dirname_basename(path):
    """ Splits the path string into directory part and file (or last dir)
    name path.

    Args:
        path (str): A path to a file or a directory.

    Returns:
        str: Directory name.
        str: Base name.
    """
    dir_name = os.path.dirname(path)
    base_name = os.path.basename(path)

    return dir_name, base_name


def rem_sshfs_prefix(pth):
    """ Removes the prefix '/Users/janbednarik/sshfs' from path name.

    Args:
        pth (str): Path

    Returns:
        str: Path without aforementioned prefix.
    """
    prefix = '/Users/janbednarik/sshfs'
    return (pth, pth[len(prefix):])[pth.startswith(prefix)]


def pycharm_add_paths():
    """ Manually adds certain paths to PYTHONPATH. A workaround
    for Pycharm's bug where, even if set, some paths are not added
    to PYTHONPATH when running a remote ssh interpreter.
    """
    import sys
    path_libigl = '/home/bednarik/programs/libigl/python'
    if path_libigl not in sys.path:
        sys.path.append(path_libigl)

### Tests.
if __name__ == '__main__':
    import jblib.unit_test as jbut

    ############################################################################
    # Test ls()
    jbut.next_test('ls()')

    path_root = '/Users/janbednarik/research/repos/jblib/jblib/tests/file_sys_tests/ls_01'

    files = ls(path_root, ignore_dot_underscore=False)
    assert(len(files) == 6)

    files = ls(path_root, exts='txt', ignore_dot_underscore=False)
    assert(len(files) == 2)

    files = ls(path_root, exts='jpg', ignore_dot_underscore=False)
    assert (len(files) == 3)

    files = ls(path_root, exts='jpg', ignore_dot_underscore=True)
    assert (len(files) == 2)

    files = ls(path_root, pattern_incl=r'c\.j', ignore_dot_underscore=False)
    assert (len(files) == 3)

    files = ls(path_root, pattern_excl='a', ignore_dot_underscore=False)
    assert (len(files) == 4)

    files = ls(path_root, exts=['txt', '.tar', 'jpg'],
               pattern_incl=r'(\.t|\.j)', pattern_excl=r'c\.t',
               ignore_dot_underscore=False)
    assert (len(files) == 6)
