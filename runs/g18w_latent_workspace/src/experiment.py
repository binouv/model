from pathlib import Path
_HERE=Path(__file__).resolve().parent
for _name in ('core1.py','core2.py','core3.py'):
    source=(_HERE/_name).read_text()
    exec(compile(source,str(_HERE/_name),'exec'),globals())
