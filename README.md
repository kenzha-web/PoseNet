## Movement Classification from Video

This project aims to classify movements captured from a video source using computer vision techniques. It utilizes Mediapipe, an open-source framework, to extract pose landmarks from frames and applies machine learning models to classify these movements in real-time or from pre-recorded videos.

### Features  
- Real-time Movement Classification: Processes live video streams to classify human movements.  
- Video File Analysis: Analyzes pre-recorded videos to classify movements.  
- Data Logging: Logs movement coordinates along with timestamps to CSV and Excel files for further analysis.  
- Graphical Visualization: Provides visual feedback by annotating videos with classified movement types.  

### Prerequisites  

Ensure you have the following installed:  
- Python 3.x  
- OpenCV  
- Mediapipe  
- Pandas  
- Scikit-learn  

Install dependencies using:

```bash
pip install -r requirements.txt
```

### Usage  
#### Real-time Movement Classification  
To classify movements from a connected webcam, run:

```bash
python movement_detection.py
```

Press q to stop the classification process.

### Video File Analysis  
To analyze movements from a video file, modify video_file_path in video_analysis.py and run:
```bash
python exam_violation_monitor.py
```

### Data Storage
Captured movement data is stored in:
- movement_classes_coordinates.csv: CSV file containing movement coordinates.  
- violation_journal.xlsx: Excel file logging classified movements and timestamps.  

### Contributing
Contributions are welcome! Please fork the repository and create a pull request with your improvements.  