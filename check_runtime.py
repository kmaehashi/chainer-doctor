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

    ### CuPy
    header('CuPy')
    cupy = None
    (cupy_status, cudnn_status, nccl_status) = ('N/A', 'N/A', 'N/A')
    cudart_version = None
    try:
        import cupy
        cupy_status = 'OK'
        cudart_version = cupy.cuda.runtime.runtimeGetVersion()
        try:
            import cupy.cuda.cudnn
            cudnn_status = 'OK'
        except Exception as e:
            cudnn_status = 'failed (optional) ({})'.format(repr(e))
        try:
            import cupy.cuda.nccl
            nccl_status = 'OK'
        except Exception as e:
            nccl_status = 'failed (optional) ({})'.format(repr(e))
    except Exception as e:
        cupy_status = 'failed ({})'.format(repr(e))

    report('Available', cupy_status)
    if cupy is not None:
        report('Available (cuDNN)', cudnn_status)
        report('Available (NCCL)', nccl_status)

        if hasattr(cupy, 'show_config'):
            report('show_config API', 'Available')
            print('')
            cupy.show_config()
            print('')
        else:
            report('show_config API',
                   'Not Available (optional) (requires v4.0.0+)')

        builtins = get_cdll('nvrtc-builtins')
        if builtins is None:
            report('NVRTC Builtins', 'Not Found')
        else:
            builtins_path = get_cdll_path(builtins.getArchBuiltins)
            report('NVRTC Builtins', 'Found ({})'.format(builtins_path))

        try:
            cupy.cuda.compiler.compile_using_nvrtc('')
            cupy_compile = 'OK'
        except Exception as e:
            cupy_compile = 'failed ({})'.format(repr(e))
        report('Compiler Test', cupy_compile)

    ### Python Modules
    header('Python Modules')
    def _report_pypkg(name, modname, pkg):
        import_msg = None
        try:
            mod = __import__(modname)
            version = '(unknown version)'
            if hasattr(mod, '__version__'):
                version = mod.__version__
            import_msg = 'importing {} from {}'.format(version, mod.__path__)
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
            if (cudart_version is not None and
                    not is_cuda_version_supported(cudart_version)):
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

    # iDeep
    _report_pypkg('iDeep', 'ideep4py', get_package('ideep4py'))



if __name__ == '__main__':
    main()
