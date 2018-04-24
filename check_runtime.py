#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ctypes
import ctypes.util
import pkg_resources
import os


def get_cdll(name):
    libname = ctypes.util.find_library(name)
    if libname is None:
        return None
    try:
        return ctypes.CDLL(libname)
    except OSError:
        return None


def get_cdll_path(func):
    libdl = get_cdll('dl')
    if libdl is None or not hasattr(libdl, 'dladdr'):
        return 'N/A'

    class Dl_info(ctypes.Structure):
        _fields_ = (
            ('dli_fname', ctypes.c_char_p),
            ('dli_fbase', ctypes.c_void_p),
            ('dli_sname', ctypes.c_char_p),
            ('dli_saddr', ctypes.c_void_p),
        )

    libdl.dladdr.argtypes = (ctypes.c_void_p, ctypes.POINTER(Dl_info))
    info = Dl_info()
    result = libdl.dladdr(func, ctypes.byref(info))
    if result == 0:
        return '(error)'
    return info.dli_fname.decode()


def get_package(name):
    try:
        return pkg_resources.get_distribution(name)
    except pkg_resources.DistributionNotFound:
        return None

def header(title):
    print('')
    print('=' * 40)
    print(title)
    print('=' * 40)

def report(title, status):
    print('{:<22}: {}'.format(title, status))


def main():
    ### Environment
    header('Environment')
    report('Current Directory', os.getcwd())
    for (k, v) in os.environ.items():
        if k.startswith('LD_') or k.startswith('DYLD_'):
            report('${}'.format(k), v)

    ### Libraries
    header('Libraries')
    cudart_version = None

    # CUDA Driver
    def _report_cuda_driver(name):
        cuda = get_cdll('cuda')
        if cuda is None:
            report(name, 'not found')
            return None
        version = ctypes.c_int()
        path = get_cdll_path(cuda.cuDriverGetVersion)
        ret = cuda.cuDriverGetVersion(ctypes.byref(version))
        version_msg = (
            str(version.value) if ret == 0 else '(ERROR {}!)'.format(ret))
        report(name, 'OK (version {} from {})'.format(
            version_msg, path))
        return version.value
    if _report_cuda_driver('CUDA Driver') is None:
        try:
            # Try to resolve via RPATH embedded in CuPy shared library.
            # Note that CUDA Runtime will also be resolved via CuPy.
            import cupy.cuda.driver
        except:
            pass
        _report_cuda_driver('CUDA Driver (via CuPy)')

    # CUDA Runtime
    def _report_cuda_runtime(name):
        cudart = get_cdll('cudart')
        if cudart is None:
            report(name, 'not found')
            return None
        version = ctypes.c_int()
        path = get_cdll_path(cudart.cudaRuntimeGetVersion)
        ret = cudart.cudaRuntimeGetVersion(ctypes.byref(version))
        version_msg = (
            str(version.value) if ret == 0 else '(ERROR {}!)'.format(ret))
        report('CUDA Runtime', 'OK (version {} from {})'.format(
            version_msg, path))
        return version.value
    cudart_version = _report_cuda_runtime('CUDA Runtime')
    if cudart_version is None:
        try:
            import cupy.cuda.runtime
        except:
            pass
        cudart_version = _report_cuda_runtime('CUDA Runtime (via CuPy)')

    # cuDNN
    def _report_cudnn(name):
        cudnn = get_cdll('cudnn')
        if cudnn is None:
            report(name, 'not found (optional)')
            return False
        path = get_cdll_path(cudnn.cudnnGetVersion)
        version = cudnn.cudnnGetVersion()
        report(name, 'OK (version {} from {})'.format(version, path))
        return True
    if not _report_cudnn('cuDNN'):
        try:
            import cupy.cudnn
            import cupy.cuda.cudnn
        except:
            pass
        _report_cudnn('cuDNN (via CuPy)')

    # NCCL
    def _report_nccl(name):
        nccl = get_cdll('nccl')
        if nccl is None:
            report(name, 'not found (optional)')
            return False
        path = get_cdll_path(nccl.ncclGetUniqueId)
        report(name, 'OK (from {})'.format(path))
        return True
    if not _report_nccl('NCCL'):
        try:
            import cupy.cuda.nccl
        except:
            pass
        _report_nccl('NCCL (via CuPy)')

    ### Python Modules
    header('Python Modules')
    def _report_pypkg(name, modname, pkg):
        import_msg = None
        try:
            mod = __import__(modname)
            import_msg = 'importing {} from {}'.format(
                mod.__version__, mod.__path__)
        except Exception as e:
            import_msg = 'import failed with {}: {}'.format(
                type(e).__name__, str(e))

        install_msg = 'not installed'
        if pkg is not None:
            install_msg = 'OK ({} version {} from {})'.format(
                pkg.project_name, pkg.version, pkg.location)
        status = '{} ({})'.format(install_msg, import_msg)

        report(name, status)

    # Chainer
    _report_pypkg('Chainer', 'chainer', get_package('chainer'))

    # CuPy
    cupy_pkgs = [
        ('cupy',        lambda v: 7000 <= v < 10000),
        ('cupy-cuda80', lambda v: 8000 <= v <  9000),
        ('cupy-cuda90', lambda v: 9000 <= v <  9100),
        ('cupy-cuda91', lambda v: 9100 <= v <  9200),
    ]
    cupy_found = None
    for (pkgname, is_cuda_version_supported) in cupy_pkgs:
        pkg = get_package(pkgname)
        if pkg is not None:
            _report_pypkg('CuPy', 'cupy', pkg)
            if not is_cuda_version_supported(cudart_version):
                print('*** ERROR: This CuPy package ({}) does not support '
                      'CUDA version {}!'.format(pkgname, cudart_version))
            if cupy_found is not None:
                print('*** ERROR: multiple CuPy packages are installed! '
                      'You can only install one of {}.'''.format(
                          [x[0] for x in cupy_pkgs]))
            cupy_found = pkg
    if cupy_found is None:
        _report_pypkg('CuPy', 'cupy', None)

    # NumPy
    _report_pypkg('NumPy', 'numpy', get_package('numpy'))


if __name__ == '__main__':
    main()
