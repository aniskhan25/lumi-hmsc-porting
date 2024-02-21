# HMSC setup on Mahti

## Installing tensorflow and dependencies

Installation done to `$HMSCDIR`.
You can set it for example to `export HMSCDIR=/scratch/project_.../$USER/hmsc`:

```bash
# Install
ml tensorflow/2.12
export PYTHONUSERBASE=$HMSCDIR/python/local
python3 -m pip install tensorflow_probability==0.20.1
python3 -m pip install pyreadr
git clone https://github.com/aniskhan25/hmsc-hpc.git $HMSCDIR/hmsc-hpc

# Create script to do module loads etc environment setup
cat << EOF > $HMSCDIR/setup-env.sh
#!/bin/bash
ml tensorflow/2.12
export HMSCDIR=\${HMSCDIR:-$HMSCDIR}
export PYTHONUSERBASE=\$HMSCDIR/python/local
export PYTHONPATH=$HMSCDIR/hmsc-hpc
export PATH=\$HMSCDIR/bin:\$PATH
EOF
chmod a+x $HMSCDIR/setup-env.sh
```

## Using the installation

### Running in an interactive session

```bash
source $HMSCDIR/setup-env.sh

srun -p gputest --nodes=1 --ntasks-per-node=1 --gres=gpu:a100:1 -t 0:15:00 --pty bash

python3 input.py

```
