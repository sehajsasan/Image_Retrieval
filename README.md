# Image_Retrieval

1. Create data dictionary using the query and gallery directory -
 In "create_dataset.ipynb":
    replace the following directories with your local paths and run the cell
        gallery_dir = '/notebooks/Image_Retrieval/human_activity_retrieval_dataset/gallery'
        query_dir = '/notebooks/Image_Retrieval/human_activity_retrieval_dataset/query_images'
        new_gallery = "/notebooks/Image_Retrieval/EasyPositiveHardNegative-master/gallery"
        new_query = "/notebooks/Image_Retrieval/EasyPositiveHardNegative-master/query"
        json_file = '/notebooks/Image_Retrieval/human_activity_retrieval_dataset/test_image_info.json'
    replace "src" with your dataset location which will be created running the above cell
        -> src='/notebooks/Image_Retrieval/EasyPositiveHardNegative-master/dataset'
    "data_dict_emb_test.pth" would be generated.
2. In _code/evaluate_EPHN.ipynb:
    replace the following directories with your local paths and run the cell
     ->dsets_dict = torch.load("/notebooks/Image_Retrieval/EasyPositiveHardNegative-master/data_dict_emb_test.pth") 
        with the "data_dict_emb_test.pth" generated from the 1st step.
     ->checkpoint = torch.load("/notebooks/Image_Retrieval/EasyPositiveHardNegative-master/_result/EPHN/HAR_R50/G16_lr0.03/model_state_dict_R5050.pth")
        with the checkpoint model provided in the Gdrive Link
3. Run the cells in order to get results and mAP scores.
    Fvec_val:     Query feature vectors, N_val by D torch.Tensor
    Fvec_gal:     Gallery feature vectors, N_gal by D torch.Tensor
    imgLab_val:   Query image label, python list
    imgLab_gal:   Gallery image label, python list
    rank:         k of mAP@k, python list

4. Results
    Results from running all the query images at k=1,10 are given as zip folder in the drive.

Checkpoint models and retrieved result images here : [Link](https://drive.google.com/drive/folders/1K1J9jJ-FaYFsYGKIL7jlPt0BgmmhuLL7?usp=sharing)
   
