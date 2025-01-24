# 🔍 Image Retrieval System - CS336.P11.KHTN

## Teacher Instruction
* PhD. Ngo Duc Thanh

## Team

| No. | Full name | Student ID | Email | Github |
| :---: | --- | --- | --- | --- |
| 1 | Trần Như Cẩm Nguyên | 22520004 | 22520004@gm.uit.edu.vn | [cnmeow](https://github.com/cnmeow) |
| 2 | Trần Thị Cẩm Giang | 22520361 | 22520361@gm.uit.edu.vn | [Yangchann](https://github.com/Yangchann) |
| 3 | Nguyễn Hữu Hoàng Long | 22520817 | 22520817@gm.uit.edu.vn | [EbisuRyu](https://github.com/EbisuRyu) |
| 4 | Đặng Hữu Phát | 22521065 | 22521065@gm.uit.edu.vn | [HuuPhat125](https://github.com/HuuPhat125) |
| 5 | Phan Hoàng Phước | 22521156 | 22521156@gm.uit.edu.vn | [HPhuoc0906](https://github.com/HPhuoc0906) |

## Introduction

This project aims to build a web application for retrieving relevant images based on textual descriptions, using some of the most powerful models such as CLIP, BLIP, BEIT. It supports two main query types:
- Text: Search for relevant images based on text input
- Image: Upload an image to find related textual descriptions.

By combining state-of-the-art deep learning models, the system ensures high accuracy and versatility for image-based search tasks.

## Features

- **Search by Text**: Enter a text query to retrieve relevant images from the database.  
  Example: *"A baby girl is wearing a red hat"*.

  <img width="1440" alt="image" src="https://github.com/user-attachments/assets/213634cb-3ff0-422a-af63-fb8c4a41e0f7" />

- **Search by Image**: Upload an image to retrieve related text descriptions, captions, or tags.
  ![image](https://github.com/user-attachments/assets/d95c3477-b473-49ec-a620-eb5785c597b5)

  Example: Upload a picture of a man
  ![image](https://github.com/user-attachments/assets/417efc31-8139-4d5b-8e81-10b5b27987d8)

  Receive query results similar to the image just uploaded
  ![image](https://github.com/user-attachments/assets/32411f71-9baf-49d8-96da-a2f5e944b2b5)

- **Powered by CLIP, BLIP, and BEIT**:
  - **CLIP**: Matches images and text by extracting semantic features.
  - **BLIP**: Automatically generates captions for images.
  - **BEiT**: Provides robust image feature extraction.

- **User-friendly Web Interface**:
  - Upload an image or enter text to perform a search.
  - Displays query results in real-time.
  
## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/cnmeow/ImageRetrievalSystem
   cd ImageRetrievalSystem
   ```
   
2. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download data**: In this project, we use [Flickr30k dataset](https://shannon.cs.illinois.edu/DenotationGraph/)
   - Download folder `images` from [https://drive.google.com/file/d/1NytawwPo2ewdPP2oQWPzvsgIjbRSe4u4](https://drive.google.com/file/d/1NytawwPo2ewdPP2oQWPzvsgIjbRSe4u4)
     - It contains images from the Flickr30k dataset which we have renamed by id to facilitate querying.
   - Download folder `data` from [https://drive.google.com/file/d/18vr_-iD8G7wdxt_xSsB2npgtuslXH_RT](https://drive.google.com/file/d/18vr_-iD8G7wdxt_xSsB2npgtuslXH_RT)
     - It contains dict files, features and weights of CLIP, BLIP, BEIT models.
   - Download folder `index` from [https://drive.google.com/file/d/1KYi4nz5uZUQf5q5zpuBbmzMR5p1bQuLb](https://drive.google.com/file/d/1KYi4nz5uZUQf5q5zpuBbmzMR5p1bQuLb)
     - It contains bin files of models with this dataset.

4. **Directory structure**:
```bash
   ImageRetrievalSystem/
    ├── app.py                  # Main application file (Flask app)
    ├── static/                 
    │   ├── flickr30k/
    │       ├── images/         # Folder images just downloaded
    ├── templates/
    ├── data/                   # Folder data just downloaded
    ├── index/                  # Folder index just downloaded
    ├── notebooks/              # Notebook model for system use, no web required
    ├── repos/  
    ├── src/                    # Source code of model
    ├── requirement.txt
```

5.  **Web application**:
- Run the web server:
```bash
flask run
```
- Open your web browser and go to [http://127.0.0.1:5000](http://127.0.0.1:5000).
  ![image](https://github.com/user-attachments/assets/7479fd7c-78f8-407c-b872-18f59d25f05b)

