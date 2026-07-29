#!/bin/bash
# End-to-end pipeline for the adjoint-based airfoil shape optimization.
# Runs INSIDE the DAFoam Docker environment, from the repository root.
set -e
NP=${NP:-4}

echo "== 1. Mesh generation =="
cd mesh
python generate_cmesh.py --case ../case --run
cd ..

echo "== 2. FFD lattice =="
python ffd/generate_ffd.py --case case

echo "== 3. Boundary conditions =="
cp -r case/0_orig case/0

cd case

echo "== 4. Baseline primal (trimmed to CL target) =="
mpirun -np $NP python run_primal.py --trim

echo "== 5. Gradient verification (directional) =="
mpirun -np $NP python verify_gradients.py --task directional

echo "== 6. GEMSEO optimization =="
mpirun -np $NP python ../mdo/run_optimization.py --max-iter 50

echo "== 7. Post-processing =="
cd ../postprocessing
python plot_history.py --db ../case/opt_history.h5
python plot_cp.py --case ../case
python plot_shapes.py --case ../case
echo "DONE."
