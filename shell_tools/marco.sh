#!bin/bash

function marco(){
	echo $(pwd) > ~/working_dir
}

function polo(){
	cd $(cat ~/working_dir)
}
