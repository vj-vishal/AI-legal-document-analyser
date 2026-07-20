import os
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

def setup_tracing(service_name: str):
    """
    Configures OpenTelemetry to send traces to Jaeger.
    Call this on startup in both FastAPI and Celery.
    """
    # 1. Define the service name so you can identify it in Jaeger
    resource = Resource.create({"service.name": service_name})

    # 2. Set up the Tracer Provider
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    # 3. Configure the Exporter to send to Jaeger via OTLP HTTP
    # Jaeger natively listens on port 4318 for OTLP HTTP
    jaeger_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
    
    otlp_exporter = OTLPSpanExporter(endpoint=jaeger_endpoint)
    
    # 4. Use a Batch processor so we don't slow down the application
    span_processor = BatchSpanProcessor(otlp_exporter)
    provider.add_span_processor(span_processor)
    
    return trace.get_tracer(service_name)