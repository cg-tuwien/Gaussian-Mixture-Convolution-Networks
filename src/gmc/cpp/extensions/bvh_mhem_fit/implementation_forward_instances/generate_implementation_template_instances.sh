#!/bin/bash


n_reduction_list="2 4 8 16"
floating_type_list="float double"
dimension_list="2 3"

for n_reduction in $n_reduction_list; do
    for floating_type in $floating_type_list; do
        for dimension in $dimension_list; do
            filename="template_instance_implementation_forward_${n_reduction}_${floating_type}_${dimension}.cu"
            if [[ -f "$filename" ]]; then
                echo "$filename exists already"
            else
                echo "bvh_mhem_fit/implementation_forward_instances/$filename"
                echo '#include "bvh_mhem_fit/implementation_forward.h"' >> $filename
                echo '' >> $filename
                echo 'namespace bvh_mhem_fit {' >> $filename
                echo "template std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl_t<$n_reduction, $floating_type, $dimension>(at::Tensor mixture, const BvhMhemFitConfig& config, unsigned n_components_target);" >> $filename
                echo '} // namespace bvh_mhem_fit' >> $filename
            fi
        done
    done
done
