import os
from typing import Optional

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

from backend.services import REVIEW_LOG_CSV, CreditFairnessService
from backend.pdf_generator import create_audit_report_pdf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_INDEX = os.path.join(BASE_DIR, "frontend", "index.html")

service: Optional[CreditFairnessService] = None


def get_service() -> CreditFairnessService:
    global service
    if service is None:
        service = CreditFairnessService()
    return service


def _parse_slice_params(request: Request) -> tuple[Optional[int], int]:
    raw_size = request.query_params.get("sample_size", "").strip()
    raw_seed = request.query_params.get("seed", "42").strip()
    sample_size = int(raw_size) if raw_size else None
    seed = int(raw_seed) if raw_seed else 42
    return sample_size, seed


async def index(request: Request) -> Response:
    if not os.path.exists(FRONTEND_INDEX):
        return PlainTextResponse("frontend/index.html not found", status_code=404)
    return FileResponse(FRONTEND_INDEX)


async def health(request: Request) -> JSONResponse:
    svc = get_service()
    return JSONResponse(svc.get_health())


async def dataset_status(request: Request) -> JSONResponse:
    svc = get_service()
    return JSONResponse(svc.get_dataset_status())


async def upload_dataset(request: Request) -> JSONResponse:
    try:
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            return JSONResponse({"error": "No CSV file uploaded under field 'file'."}, status_code=400)
        raw_bytes = await upload.read()
        filename = getattr(upload, "filename", "uploaded.csv")
        svc = get_service()
        summary = svc.load_dataset_from_bytes(filename=filename, raw_bytes=raw_bytes)
        return JSONResponse(summary)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def load_default_dataset(request: Request) -> JSONResponse:
    try:
        svc = get_service()
        return JSONResponse(svc.load_default_dataset())
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def train_model(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            payload = {}
        svc = get_service()
        metrics = svc.train_and_evaluate(payload)
        return JSONResponse({
            "training_metrics": metrics,
            "overview": svc.get_overview(),
        })
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def overview(request: Request) -> JSONResponse:
    try:
        sample_size, seed = _parse_slice_params(request)
        svc = get_service()
        return JSONResponse(svc.get_overview(sample_size=sample_size, seed=seed))
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def fairness(request: Request) -> JSONResponse:
    try:
        group_by = request.query_params.get("group_by", "CODE_GENDER")
        sample_size, seed = _parse_slice_params(request)
        svc = get_service()
        return JSONResponse(svc.compute_fairness(group_col=group_by, sample_size=sample_size, seed=seed))
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def applicant_detail(request: Request) -> JSONResponse:
    try:
        row_index = int(request.path_params["row_index"])
        sample_size, seed = _parse_slice_params(request)
        svc = get_service()
        return JSONResponse(svc.explain_test_applicant(row_index=row_index, sample_size=sample_size, seed=seed))
    except IndexError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def sample_applicant(request: Request) -> JSONResponse:
    try:
        flagged_param = request.query_params.get("flagged", "").strip()
        flagged = int(flagged_param) if flagged_param in ("0", "1") else None
        sample_size, seed = _parse_slice_params(request)
        svc = get_service()
        idx = svc.find_sample_row_index(flagged=flagged, sample_size=sample_size, seed=seed)
        return JSONResponse(svc.explain_test_applicant(row_index=idx, sample_size=sample_size, seed=seed))
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def predict_custom(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            return JSONResponse({"error": "JSON object body required"}, status_code=400)
        svc = get_service()
        return JSONResponse(svc.predict_custom_applicant(payload))
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def list_reviews(request: Request) -> JSONResponse:
    svc = get_service()
    return JSONResponse({"reviews": svc.get_reviews()})


async def create_review(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        svc = get_service()
        record = svc.save_review(
            row_index=payload.get("row_index"),
            probability=float(payload.get("probability", 0.0)),
            advisory_flag=int(payload.get("advisory_flag", 0)),
            human_decision=str(payload.get("human_decision", "Hold for Verification")),
            note=str(payload.get("note", "")),
            applicant_details=payload.get("applicant_details"),
        )
        return JSONResponse({"saved": record, "reviews": svc.get_reviews()})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)



async def download_report(request: Request) -> Response:
    try:
        row_index = int(request.query_params.get("row_index", "0"))
        fmt = request.query_params.get("format", "pdf").lower()
        sample_size, seed = _parse_slice_params(request)
        svc = get_service()

        if fmt == "pdf":
            overview = svc.get_overview(sample_size, seed)
            gender = svc.compute_fairness("CODE_GENDER", sample_size, seed)
            age = svc.compute_fairness("AGE_GROUP", sample_size, seed)
            applicant = svc.explain_test_applicant(row_index, sample_size, seed)
            pdf_bytes = create_audit_report_pdf(overview, gender, age, applicant, row_index)
            return Response(
                pdf_bytes,
                media_type="application/pdf",
                headers={"Content-Disposition": "attachment; filename=PS173_Credit_Fairness_Report.pdf"},
            )

        report_text = svc.build_audit_report(row_index=row_index, sample_size=sample_size, seed=seed, fmt=fmt)
        if fmt == "txt":
            return PlainTextResponse(
                report_text,
                headers={"Content-Disposition": "attachment; filename=PS173_Credit_Fairness_Report.txt"},
            )
        return Response(
            report_text,
            media_type="text/markdown",
            headers={"Content-Disposition": "attachment; filename=PS173_Credit_Fairness_Report.md"},
        )
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


routes = [
    Route("/", index, methods=["GET"]),
    Route("/api/health", health, methods=["GET"]),
    Route("/api/dataset/status", dataset_status, methods=["GET"]),
    Route("/api/dataset/upload", upload_dataset, methods=["POST"]),
    Route("/api/dataset/default", load_default_dataset, methods=["POST"]),
    Route("/api/train", train_model, methods=["POST"]),
    Route("/api/overview", overview, methods=["GET"]),
    Route("/api/fairness", fairness, methods=["GET"]),
    Route("/api/applicant/sample", sample_applicant, methods=["GET"]),
    Route("/api/applicant/{row_index:int}", applicant_detail, methods=["GET"]),
    Route("/api/predict", predict_custom, methods=["POST"]),
    Route("/api/reviews", list_reviews, methods=["GET"]),
    Route("/api/reviews", create_review, methods=["POST"]),
    Route("/api/report", download_report, methods=["GET"]),
]

app = Starlette(routes=routes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
