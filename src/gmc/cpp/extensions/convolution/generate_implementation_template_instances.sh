#!/bin/bash

file_ending=$1
if [ -z "$1" ]; then
    file_ending='cu'
fi

n_reduction_list="1 "
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
                    echo "convolution/$filename exists already"
                else
                    echo "convolution/$filename"
                    if [ "$floating_type" == "double" ]; then
                        echo "#ifndef GPE_ONLY_FLOAT" >> $filename
                    fi
                    if [ "$dimension" == "3" ]; then
                        echo "#ifndef GPE_ONLY_2D" >> $filename
                    fi
                    
                    echo "#include \"convolution/implementation_${direction}.h\"" >> $filename
                    echo "#include <utility>" >> $filename
                    echo '' >> $filename
                    echo 'namespace convolution {' >> $filename
                    
                    if [ "${direction}" == "forward" ]; then
                        echo "template torch::Tensor forward_impl_t<$floating_type, $dimension>(const torch::Tensor& data, const torch::Tensor& kernels);" >> $filename
                    else
                        echo "template std::pair<torch::Tensor, torch::Tensor> backward_impl_t<$floating_type, $dimension>(const torch::Tensor& grad, const torch::Tensor& data, const torch::Tensor& kernels);" >> $filename
                    fi
                    
                    echo '} // namespace convolution' >> $filename
                    
                    if [ "$dimension" == "3" ]; then
                        echo "#endif // GPE_ONLY_2D" >> $filename
                    fi
                    if [ "$floating_type" == "double" ]; then
                        echo "#endif // GPE_ONLY_FLOAT" >> $filename
                    fi
                fi
            done
        done
    done
done
