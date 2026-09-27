from pathlib import Path
_HERE=Path(__file__).resolve().parent
exec(compile((_HERE/'study_core1.py').read_text(),str(_HERE/'study_core1.py'),'exec'),globals())
exec(compile((_HERE/'study_core2.py').read_text(),str(_HERE/'study_core2.py'),'exec'),globals())
