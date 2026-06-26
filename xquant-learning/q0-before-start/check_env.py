import sys
import shutil

REQUIRED_PYTHON = (3, 12)

REQUIRED_PACKAGES = [("jupyter", "jupyter"),
                     ("pandas", "pandas"),
                     ("numpy", "numpy"),
                     ("matplotlib", "matplotlib"),
                     ("akshare", "akshare"),
                     ("yfinance", "yfinance"),
                     ("oxq", "open-xquant"),
                    ]

AKSHARE_TEST_FUNC = "fund_etf_hist_em"
AKSHARE_TEST_KWARGS = {"symbol": "510300", "period": "daily", "adjust": "qfq"}

OPEN_XQUANT_INSTALL = 'uv pip install "open-xquant @ git+https://github.com/xingwudao/open-xquant.git"'

def check_python_version():
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= REQUIRED_PYTHON:
        return True, f"Python {version_str}" 
    return False, f"Python {version_str}（需要 >= {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}）"

def check_venv():
    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        return True, "虚拟环境已激活"
    return False, "未检测到虚拟环境，请先运行 source .venv/bin/activate"

def check_package(import_name, display_name):
    try:
        __import__(import_name)
        return True, display_name
    except ImportError:
        if display_name == "open-xquant":
            return False, f"{display_name} — 未安装，请运行: {OPEN_XQUANT_INSTALL}"
        return False, f"{display_name} — 未安装，请运行: uv pip install {display_name}"

def check_jupyter_command():
    if shutil.which("jupyter"):
        return True, "jupyter 命令可用"
    return False, "jupyter 命令不可用，请运行: uv pip install jupyter"

def check_data_source():
    try:
        import akshare as ak
        func = getattr(ak, AKSHARE_TEST_FUNC)
        df = func(**AKSHARE_TEST_KWARGS)
        if len(df) > 0:
            return True, f"akshare 数据源正常（获取到 {len(df)} 条数据）"
        return False, "akshare 返回了空数据，可能是接口变动"
    except ImportError:
        return False, "akshare 未安装，跳过数据源测试"
    except Exception as e:
        return False, f"akshare 数据获取失败: {e}"

def main():
    print()
    print("=" * 50)
    print(" XQuant 课程环境检查")
    print("=" * 50)
    print()
    results = []

    ok, msg = check_python_version()
    results.append((ok, msg))
    
    ok, msg = check_venv()
    results.append((ok, msg))
    
    for import_name, display_name in REQUIRED_PACKAGES:
        ok, msg = check_package(import_name, display_name)
        results.append((ok, msg))
    
    ok, msg = check_jupyter_command()
    results.append((ok, msg))
    
    ok, msg = check_data_source()
    results.append((ok, msg))
    
    passed = 0
    failed = 0
    for ok, msg in results:
        if ok:
            print(f"  [OK]   {msg}")
            passed += 1
        else:
            print(f"  [FAIL] {msg}")
            failed += 1
    
    print()
    print("-" * 50)
    
    if failed == 0:
        print(f"  结果: {passed}/{passed} 全部通过!")
        print()
        print("  环境配置完成，可以开始课程了。")
    else:
        print(f"  结果: {passed}/{passed + failed} 通过，{failed} 项需要修复")
        print()
        print("  请把上面的输出结果发给 AI 助手，让它帮你修复失败项。")
    
    print()
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
