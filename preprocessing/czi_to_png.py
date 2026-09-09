# ===== Third-Party Imports =====
import aicspylibczi
import numpy as np
# ===== Own Modules =====
from settings import setting
import preprocessing.functions as fn

########
# MAIN #
########

# Convert Zeiss .czi mosaic files to individual training images.
def main() -> None:
    # Create program folders
    fn.create_prg_folders()

    # Get a list with all czi files in input/ folder
    czi_list = fn.get_czi_file_list()

    if not czi_list:
        print("No .czi files found in input/ folder!")
        return

    # Iterate over file list
    for file_name in czi_list:
        print(f"\n>> PROCESSING IMAGE {file_name}:")

        ###################
        # Load czi mosaic #
        ###################

        file_name_ext = file_name + setting['preproc_czi_img_ext']
        img_data = aicspylibczi.CziFile(setting['pth_input'] / file_name_ext)
        print(f"Loading of mosaic .czi image {file_name_ext} successful.")

        # Create directory for each czi image
        fn.create_export_folder(file_name)

        #################
        # Read z-planes #
        #################

        print(f"Prepare image for slicing. Please wait...")

        if setting['preproc_sharpest_z_plane'] is None:
            num_z_planes = fn.read_num_z_planes(img_data)
            z_planes = []
            for z_plane in range(num_z_planes):
                img = fn.read_czi_mosaic(img_data, channel=0, z_plane=z_plane)
                img = np.squeeze(img)
                z_planes.append(img)
                del img
        else:
            num_z_planes = 1
            img = fn.read_czi_mosaic(img_data, channel=0, z_plane=setting['preproc_sharpest_z_plane'])
            img = np.squeeze(img)
            z_planes = [img]
            del img

        print(f"Image is ready for slicing.")

        ################
        # Slice images #
        ################

        num_tiles_x = setting['preproc_num_tiles']['x']
        num_tiles_y = setting['preproc_num_tiles']['y']

        for i in range(num_tiles_y):
            for j in range(num_tiles_x):
                slice_origin = fn.get_slice_origin(img_data, i, j, num_tiles_x)
                print(f"> Processing image slice {i}_{j}...")

                z_slices = []
                for n in range(num_z_planes):
                    sliced_np = fn.slice_np_img(z_planes[n], slice_origin)
                    z_slices.append(sliced_np)

                if setting['preproc_sharpest_z_plane'] is None:
                    sharpest_image, sharpest_index = fn.find_sharpest_plane(z_slices, fn.tenengrad_sharpness)
                    print(f"Sharpest image of z-stack is from plane {sharpest_index}.")
                else:
                    sharpest_image = z_slices[0]
                    sharpest_index = setting['preproc_sharpest_z_plane']
                    print(f"Sharpest plane of z-stack was set as plane {sharpest_index}.")

                ####################
                # Image processing #
                ####################

                for idx, img in enumerate(z_slices):
                    z_slices[idx] = fn.reduce_noise(img)

                sharpest_image = fn.percentile_normalization(sharpest_image)
                sharpest_image = fn.convert_to_8bit(sharpest_image)
                sharpest_image = fn.pil_from_np(sharpest_image)
                sharpest_image = fn.resize_pil_img(sharpest_image)

                ##############
                # Save Image #
                ##############

                sharpest_image_name = f'{file_name}_y{i}_x{j}_z{sharpest_index}.png'
                sharpest_image.save(setting['pth_output'] / file_name / sharpest_image_name)

        print(f">> PROCESSING OF IMAGE {file_name} FINISHED!")


if __name__ == "__main__":
    main()