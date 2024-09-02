# LCN
Logical Credal Networks (LCNs)

## Installation Instructions
For development and running experiments, we need to create a Python 3.10 
environment that contains the required dependencies. This can be done easily by
cloning the `LCN` git repository, creating a conda environment with Python 3.10
and installing the `LCN` package in that environment, as follows:

```
git clone git@github.ibm.com:IBM-Research-AI/LCN.git
cd LCN
conda create -n lcn python=3.10
conda activate lcn
pip install -e .
```

## Installing ipopt on Linux

The LCN solver requires the non-linear solver `ipopt`. To install the 
solver on Linux, you can use the `coinbrew` tool. Simply download the `coinbrew`
script from https://coin-or.github.io/coinbrew/ (make sure to also run `chmod u+x coinbrew`).
`coinbrew` automates the download of the source code for ASL, MUMPS, and Ipopt 
and the sequential build and installation of these three packages.

### Install required dependencies on Linux:
* Ubuntu: `sudo apt-get install gcc g++ gfortran git cmake liblapack-dev pkg-config --install-recommends`
* Fedora: `sudo yum install gcc gcc-c++ gcc-gfortran git cmake lapack-devel`

After obtaining the `coinbrew` script, run:

```
/path/to/coinbrew fetch Ipopt --no-prompt
/path/to/coinbrew build Ipopt --prefix=/dir/to/install --test --no-prompt --verbosity=3
/path/to/coinbrew install Ipopt --no-prompt
```

Also, add the following to your `.bashrc` file:

```
export LD_LIBRARY_PATH=/dir/to/install/lib
export PATH="/dir/to/install/bin:$PATH"
```

## Installing ipopt on MacOS

To install `ipopt` on MacOS just run `brew install ipopt`
