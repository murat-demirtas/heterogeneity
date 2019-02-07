filename="model_optimization.py"
model_name="heterogeneous"
n_particles=5
n_tasks=5
n_iterations=25
output_directory="heterogeneous"

for iter in {1..25}
do
for samplers in {0..4}
do
python $filename $model_name $n_particles $n_tasks $samplers sampler $output_directory &
done
wait
python $filename $model_name $n_particles $n_tasks 0 wrapper $output_directory
done

