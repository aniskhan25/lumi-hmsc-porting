# HMSC setup on LUMI

## Installing tensorflow and dependencies

Installation done to `$WORKDIR`.
You can set it for example to `export WORKDIR=/scratch/project_.../$USER`:

```bash
ml use /appl/local/csc/modulefiles
ml tensorflow/2.12
export PYTHONUSERBASE=$WORKDIR/python/local
python3 -m pip install tensorflow_probability==0.20.0

# Create source script
cat << EOF > $WORKDIR/setup-env.sh
#!/bin/bash
ml use /appl/local/csc/modulefiles
ml tensorflow/2.12
export WORKDIR=\${WORKDIR:-$WORKDIR}
export PYTHONUSERBASE=\$WORKDIR/python/local
export PATH=\$WORKDIR/bin:\$PATH
EOF
chmod a+x $WORKDIR/setup-env.sh
```

## Using the installation

### Running in an interactive session

```bash
source $WORKDIR/setup-env.sh

srun -p small-g --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -t 1:00:00 --pty bash

python3 input.py

```

### Using tensorboard

Launch tensorboard on a login node:
```bash
source $WORKDIR/setup-env.sh

python3 -m tensorboard.main --logdir=logdir --port=$(($(id -u) % (65536 - 4096) + 4096))

```

Check the port number and set up ssh port forwarding on the local machine
(use the correct port and login node numbers):
```bash
ssh -L 6006:localhost:XXXXX lumi-uanXX.csc.fi
```

Open tensorboard at http://localhost:6006/#profile

