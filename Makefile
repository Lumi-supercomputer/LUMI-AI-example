
build:
	./install_venv.sh

download:
	./get_data.sh

convert:
	sbatch convert.sh && sbatch scripts/squashfs/convert_to_squashfs.sh
