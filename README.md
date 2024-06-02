# Image Retrieval

## Instructions

### 1. Create Data Dictionary

In `create_dataset.ipynb`:
- Replace the following directories with your local paths and run the cell:
    ```python
    gallery_dir = '/notebooks/Image_Retrieval/human_activity_retrieval_dataset/gallery'
    query_dir = '/notebooks/Image_Retrieval/human_activity_retrieval_dataset/query_images'
    new_gallery = '/notebooks/Image_Retrieval/EasyPositiveHardNegative-master/gallery'
    new_query = '/notebooks/Image_Retrieval/EasyPositiveHardNegative-master/query'
    json_file = '/notebooks/Image_Retrieval/human_activity_retrieval_dataset/test_image_info.json'
    ```
- Replace `src` with your dataset location which will be created by running the above cell:
    ```python
    src='/notebooks/Image_Retrieval/EasyPositiveHardNegative-master/dataset'
    ```
- A file named `data_dict_emb_test.pth` will be generated.

### 2. Evaluate EPHN

In `_code/evaluate_EPHN.ipynb`:
- Replace the following directories with your local paths and run the cell:
    Replace with the `data_dict_emb_test.pth` generated from the first step:
    ```python
    dsets_dict = torch.load('/notebooks/Image_Retrieval/EasyPositiveHardNegative-master/data_dict_emb_test.pth')
    ```
    Replace with the checkpoint model provided in the Google Drive link.
    ```python
    checkpoint = torch.load('/notebooks/Image_Retrieval/EasyPositiveHardNegative-master/_result/EPHN/HAR_R50/G16_lr0.03/model_state_dict_R5050.pth')
    ```
   

### 3. Run Cells in _code/evaluate_EPHN.ipynb to Get Results

Run the cells in order to get results and mAP scores:
- `Fvec_val`: Query feature vectors, `N_val` by `D` `torch.Tensor`
- `Fvec_gal`: Gallery feature vectors, `N_gal` by `D` `torch.Tensor`
- `imgLab_val`: Query image label, `python list`
- `imgLab_gal`: Gallery image label, `python list`
- `rank`: k of mAP@k, `python list`

### 4. Results

Results from running all the query images at k=1,10 are provided as a zip folder in the drive.

Checkpoint models and retrieved result images are available here: [Google Drive Link](https://drive.google.com/drive/folders/1K1J9jJ-FaYFsYGKIL7jlPt0BgmmhuLL7?usp=sharing)