from pathlib import Path
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.predictor import predict_ecg


app = FastAPI(
    title="Heart AI API",
    description="AI-powered cardiac analysis backend",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Heart AI Backend is running!"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Heart AI"
    }


@app.post("/predict")
async def predict(
    dat_file: UploadFile = File(...),
    hea_file: UploadFile = File(...)
):

    try:

        # --------------------------------------------------
        # Validate file types
        # --------------------------------------------------

        if not dat_file.filename.lower().endswith(".dat"):
            raise HTTPException(
                status_code=400,
                detail="First file must be a PTB-XL .dat ECG file."
            )

        if not hea_file.filename.lower().endswith(".hea"):
            raise HTTPException(
                status_code=400,
                detail="Second file must be the matching PTB-XL .hea header file."
            )


        # --------------------------------------------------
        # Make sure both files belong to the same ECG
        # --------------------------------------------------

        dat_name = Path(
            dat_file.filename
        ).stem

        hea_name = Path(
            hea_file.filename
        ).stem


        if dat_name != hea_name:

            raise HTTPException(
                status_code=400,
                detail="The .dat and .hea files must belong to the same ECG record."
            )


        # --------------------------------------------------
        # Temporary directory
        # --------------------------------------------------

        temp_dir = Path("temp_ecg")

        temp_dir.mkdir(
            exist_ok=True
        )


        # --------------------------------------------------
        # Save .dat
        # --------------------------------------------------

        dat_path = temp_dir / dat_file.filename


        with open(
            dat_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                dat_file.file,
                buffer
            )


        # --------------------------------------------------
        # Save .hea
        # --------------------------------------------------

        hea_path = temp_dir / hea_file.filename


        with open(
            hea_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                hea_file.file,
                buffer
            )


        # --------------------------------------------------
        # Remove extension
        #
        # WFDB expects:
        #
        # temp_ecg/00001_lr
        #
        # and automatically loads:
        #
        # 00001_lr.dat
        # 00001_lr.hea
        # --------------------------------------------------

        record_path = temp_dir / dat_name


        # --------------------------------------------------
        # Predict
        # --------------------------------------------------

        predictions = predict_ecg(
            record_path
        )


        # --------------------------------------------------
        # Return result
        # --------------------------------------------------

        return {
            "status": "success",
            "filename": dat_file.filename,
            "predictions": predictions
        }


    except HTTPException:

        raise


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )