#!/bin/bash


run () {
	local file_in="./data/input/$1.mp4"
	local file_out="./data/output/$1-$2.mp4"
	local config="./perturb_config.yaml"
	local clean_arg="$(echo $2 | tr _ -)"

	local my_cmd="core/venv/bin/python perturb.py -c $config -i $file_in -o $file_out --$clean_arg"
	echo "Evaluating --$clean_arg for $file_in to $file_out ..."
	$my_cmd
}


operate () {
	for i in $(seq -f "%02g" 1 5)
	do
		run $i $1 &
	done
	wait

	for i in $(seq -f "%02g" 6 10)
	do
		run $i $1 &
	done

	wait
	echo "Done."
}


operate $1
