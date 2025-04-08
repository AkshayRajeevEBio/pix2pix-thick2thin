import argparse
import pandas as pd
from tqdm import tqdm
import json

def main(args):
    if False:
        v3 = pd.read_csv(args.v2, low_memory=False)
        v41 = pd.read_csv(args.v41, low_memory=False)
        v42 = pd.read_csv(args.v42, low_memory=False)
        ncm = pd.read_csv(args.ncm, low_memory=False)
        
        latest_v4 = pd.concat([
            v41.loc[~v41.original_PatientID.isin(v42.original_PatientID)],
            v42
            ], ignore_index=True
        )
        common_v4_pts = set(latest_v4.original_PatientID.unique()).intersection(
            set(ncm.original_PatientID.unique())
        )
        remaining_common_cases = set(ncm.original_PatientID.unique()) - \
            set(latest_v4.original_PatientID.unique())
        thick_wall_df = pd.concat([
            latest_v4.loc[latest_v4.original_PatientID.isin(common_v4_pts)],
            v3.loc[v3.original_PatientID.isin(remaining_common_cases)]
        ])
        del v3, v41, v42, latest_v4
        
        thick_wall_df.to_csv('/inst/rajeev/pix2pix/thick_wall_gmd.csv',
                             index=False)
        ncm.to_csv('/inst/rajeev/pix2pix/thin_wall_gmd.csv',
                   index=False)
        thin_wall_df = ncm
    if True:
        print('Loading_data..')
        thick_wall_df= pd.read_csv('/inst/rajeev/pix2pix/thick_wall_gmd.csv',
                                   low_memory=False)
        thin_wall_df = pd.read_csv('/inst/rajeev/pix2pix/thin_wall_gmd.csv',
                                   low_memory=False)

    image_map = {"healthy":{}, "unhealthy":{}}
    for pt in tqdm(thin_wall_df.original_PatientID.unique()):
        thick_wall = thick_wall_df.loc[thick_wall_df.original_PatientID == pt]
        thin_wall = thin_wall_df.loc[thin_wall_df.original_PatientID == pt]
        
        pt_images_map = pd.merge(thick_wall, thin_wall, 
            on=['location', 'site'], how='inner')
        pt_images_map.drop_duplicates(subset=['location', 'site'],
                                      inplace=True)
        pt_images_map['total_plaque'] = pt_images_map['CALCArea_x'] + \
             pt_images_map['NonCALCArea_x'] +  pt_images_map['LRNCArea_x']
        healthy_images_map_df = pt_images_map.loc[
            pt_images_map.total_plaque == 0]
        unhealthy_images_map_df = pt_images_map.loc[
            pt_images_map.total_plaque > 0]
        healthy_pt_images_d = dict(zip(
            healthy_images_map_df['section_image_path_x'],
            healthy_images_map_df['section_image_path_y'])
        )
        unhealthy_pt_images_d = dict(zip(
            unhealthy_images_map_df['section_image_path_x'],
            unhealthy_images_map_df['section_image_path_y'])
        )
        image_map['healthy'].update(healthy_pt_images_d)
        image_map['unhealthy'].update(unhealthy_pt_images_d)

    with open(args.out_path, "w") as file:
        json.dump(image_map, file, indent=4)
    print('Image_map saved to', args.out_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Curate data for Pix2Pix model.")
    parser.add_argument("--v2", type=str,
        default='/inst/rajeev/data_pipeline_testing/'\
            'vessel_tree_data_20250128-010729.csv')
    parser.add_argument("--v41", type=str,
        default='/inst/rajeev/data_pipeline_testing/'\
            'V4.1-vessel_tree_data_20250317-144003.csv')
    parser.add_argument("--v42", type=str,
        default='/inst/rajeev/data_pipeline_testing/'\
            'V4.2-vessel_tree_data_20250327-133141.csv')
    parser.add_argument("--ncm", type=str,
        default='/inst/rajeev/data_pipeline_testing/'\
            'NCM-vessel_tree_data_20250328-084011.csv')
    parser.add_argument("--out_path", type=str,
        default="/inst/rajeev/pix2pix/image_map.json")

    args = parser.parse_args()
    main(args)
