mkdir color_nbg
mkdir color_noise
mkdir gray
mkdir gray_nbg
mkdir gray_noise
mkdir color_shift
mkdir color_shift_noise
mkdir backgrounds
mkdir samples
mkdir dice_snaps

tar -xzvf data_archives/60k_color_250x250.tar.gz

python make_dice.py
python make_samples.py