# run all scripts in one script
# Usage: ./run_all.sh
# PIC_USE=19picsuse
# SAVE_FOLDER=pic19-0813 # save folder, all subfolders will be created in this folder
# JSON_FILE=8.13/all/result19-all.json
# TARGET_FOLDER=${SAVE_FOLDER}/pic19gt                  # save the results before cutting train val test
# TARGET_FOLDER_AFTER_SPLIT=${SAVE_FOLDER}/pic19gtsplit # save the results after splitting train val test
# ZOOM=19

# PIC_USE=picuse20
# SAVE_FOLDER=pic20-0813 # save folder, all subfolders will be created in this folder
# JSON_FILE=8.13/all/result20-all.json
# TARGET_FOLDER=${SAVE_FOLDER}/pic20gt                  # save the results before cutting train val test
# TARGET_FOLDER_AFTER_SPLIT=${SAVE_FOLDER}/pic20gtsplit # save the results after splitting train val test
# ZOOM=20
PIC_USE=pic1408
SAVE_FOLDER=project1408
JSON_FILE=satellite_lane_anno.json
TARGET_FOLDER=${SAVE_FOLDER}/proj19gt
TARGET_FOLDER_AFTER_SPLIT=${SAVE_FOLDER}/proj19gtsplit
ZOOM=19

if [ "$ZOOM" -eq 19 ]; then
    CLIP_SIZE=512
    IMAGE_SIZE=2048
elif [ "$ZOOM" -eq 20 ]; then
    CLIP_SIZE=1024
    IMAGE_SIZE=4096
fi

echo "start running all scripts"
echo ""

# 1. run genertae_GT_multiclass_only-multi.py to get the ground truth
echo "1. run genertae_GT_multiclass_only-multi.py to get the ground truth"
python tools/generate_GT_multiclass_only-multi.py --file_name $JSON_FILE \
    --target_folder $TARGET_FOLDER --source_folder $PIC_USE --image_size $IMAGE_SIZE
echo ""
# 2. run python generate_GT_direction_tag.py
echo "2. run python tools/generate_GT_direction_tag-multi.py"
python tools/generate_GT_direction_tag-multi.py --file_name $JSON_FILE \
    --target_folder $TARGET_FOLDER --source_folder $PIC_USE --image_size $IMAGE_SIZE
echo ""
# 3. generate train val test pics
echo "3. generate train val test pics"
python tools/train_val_test_pic.py --zoom $ZOOM --source_folder $PIC_USE/use --target_folder ${PIC_USE}trainvaltest
echo ""
4. generate train val test tag GT
echo "4. generate train val test tag GT"
python tools/train_val_test_tag.py --source_folder $TARGET_FOLDER-mask-tag/use --target_folder $TARGET_FOLDER_AFTER_SPLIT \
    --pic_folder ${PIC_USE}trainvaltest --zoom $ZOOM
echo ""
# 5. split other GT train val test
echo "5. split other GT train val test"
python tools/train_val_test_gt.py --source_folder $TARGET_FOLDER --target_folder $TARGET_FOLDER_AFTER_SPLIT \
    --pic_folder ${PIC_USE}trainvaltest --zoom $ZOOM
echo ""
# 6. cut images and GT into small pics
echo "6. cut images and GT"
bash tools/run_cut_satellite.sh $TARGET_FOLDER_AFTER_SPLIT ${PIC_USE}trainvaltest $CLIP_SIZE
echo ""
# 7. check the existence of npy files
echo "7. check the existence of npy files"
bash tools/check_npy.sh $TARGET_FOLDER_AFTER_SPLIT-cut
echo ""
# 8. find all black images
echo "8. find all black images"
bash tools/run_black_images.sh $TARGET_FOLDER_AFTER_SPLIT-cut ${PIC_USE}trainvaltest-cut
echo ""
# 9. remove all black images
echo "9. remove all black images"
bash tools/remove_black_image.sh ${PIC_USE}trainvaltest-cut/txt $TARGET_FOLDER_AFTER_SPLIT-cut \
    ${PIC_USE}trainvaltest-cut
echo ""
# 10. move used things to a new path
echo "10. move used things to a new path"
mkdir -p ${SAVE_FOLDER}/final
mv ${PIC_USE}trainvaltest-cut ${SAVE_FOLDER}/final
mv ${PIC_USE}trainvaltest ${SAVE_FOLDER}
mv $TARGET_FOLDER_AFTER_SPLIT-cut ${SAVE_FOLDER}/final
echo "all scripts have been run"
