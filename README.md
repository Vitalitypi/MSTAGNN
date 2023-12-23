# train steps

### 1、generate data

```bash
cd generate
python generate.py
```

### 2、training

```bash
python -u main.py 2>&1 | tee exps/PEMS04.log
```

