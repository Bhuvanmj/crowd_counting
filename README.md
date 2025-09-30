# Crowd Counting Dashboard

A real-time crowd counting and density visualization tool built with **CSRNet** and **Streamlit**.  
Upload videos or images to analyze crowd density and receive alerts when crowd counts exceed a threshold.



## Features

- Predict crowd density and count in images or videos.
- Visualize density maps with clear hot spots.
- Real-time alerts for overcrowding.
- Interactive and user-friendly Streamlit interface.



## Project Structure

crowd_counting_dashboard/
│
├─ src/
│ ├─ streamlit_app.py # Main Streamlit app
│ └─ best_csrnet.pth # Pre-trained CSRNet model
├─ requirements.txt # Python dependencies
├─ README.md # Project description
└─ .gitignore # Files/folders to ignore



## Installation

1. Clone the repository.
2. (Optional) Create and activate a virtual environment.
3. Install the dependencies listed in `requirements.txt`.



## Usage

- Run the Streamlit app (`streamlit_app.py`) from the `src/` folder.
- Upload a video (MP4) or an image to process.
- Adjust parameters in the sidebar:
  - Crowd threshold
  - Frame interval (process every Nth frame)
- View:
  - Real-time crowd count
  - Density map visualization
  - Alerts for overcrowding



## Notes

- Place `best_csrnet.pth` inside the `src/` folder.
- Update paths in `streamlit_app.py` if using a different directory structure.
- Supports GPU (CUDA) if available; otherwise, it defaults to CPU.

