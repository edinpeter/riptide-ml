mkdir -p data/training_noise
mkdir -p data/dice_snaps
mkdir -p results

cp ../../synthetic_images/color_shift_noise/[0-2]* data/training_noise
cp ../../synthetic_images/color_shift_noise/[3-4]* data/training_noise
cp ../../synthetic_images/color_shift_noise/[5-6]* data/training_noise

cp ../../synthetic_images/dice_snaps/[0-2]* data/dice_snaps
cp ../../synthetic_images/dice_snaps/[3-4]* data/dice_snaps
cp ../../synthetic_images/dice_snaps/[5-6]* data/dice_snaps
