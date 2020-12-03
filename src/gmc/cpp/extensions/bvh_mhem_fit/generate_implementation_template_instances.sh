#!/bin/bash

file_ending=$1
if [ -z "$1" ]; then
    file_ending='cu'
fi

n_reduction_list="2 4 8 16"
floating_type_list="float double"
dimension_list="2 3"
direction_list="forward backward"

for direction in $direction_list; do
    parent_dir="implementation_${direction}_instances"
    mkdir -p $parent_dir
    for n_reduction in $n_reduction_list; do
        for floating_type in $floating_type_list; do
            for dimension in $dimension_list; do
                filename="${parent_dir}/template_instance_implementation_${direction}_${n_reduction}_${floating_type}_${dimension}.${file_ending}"
                if [[ -f "$filename" ]]; then
                    echo "$filename exists already"
                else
                    echo "bvh_mhem_fit/$filename"
                    if [ "$n_reduction" == "8" ] || [ "$n_reduction" == "16" ]; then
                        echo "#ifndef GPE_LIMIT_N_REDUCTION" >> $filename
                    fi
                    if [ "$floating_type" == "double" ]; then
                        echo "#ifndef GPE_ONLY_FLOAT" >> $filename
                    fi
                    if [ "$dimension" == "3" ]; then
                        echo "#ifndef GPE_ONLY_2D" >> $filename
                    fi
                    
                    echo "#include \"bvh_mhem_fit/implementation_${direction}.h\"" >> $filename
                    echo '' >> $filename
                    echo 'namespace bvh_mhem_fit {' >> $filename
                    
                    if [ "${direction}" == "forward" ]; then
                        echo "template ForwardOutput forward_impl_t<$n_reduction, $floating_type, $dimension>(torch::Tensor mixture, const BvhMhemFitConfig& config);" >> $filename
                    else
                        echo "template torch::Tensor backward_impl_t<$n_reduction, $floating_type, $dimension>(torch::Tensor grad, const ForwardOutput& forward_out, const BvhMhemFitConfig& config);" >> $filename
                    fi
                    
                    echo '} // namespace bvh_mhem_fit' >> $filename
                    
                    if [ "$dimension" == "3" ]; then
                        echo "#endif // GPE_ONLY_2D" >> $filename
                    fi
                    if [ "$floating_type" == "double" ]; then
                        echo "#endif // GPE_ONLY_FLOAT" >> $filename
                    fi
                    if [ "$n_reduction" == "8" ] || [ "$n_reduction" == "16" ]; then
                        echo "#endif // GPE_LIMIT_N_REDUCTION" >> $filename
                    fi
                fi
            done
        done
    done
done
