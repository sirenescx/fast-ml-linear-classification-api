import grpc
from concurrent import futures

from fast_ml_linear_classification_models.proto.compiled import linear_classification_pb2_grpc
from fast_ml_linear_classification_models.services.linear_classificator import LinearClassificationService


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    linear_classification_pb2_grpc.add_LinearRegressionServiceServicer_to_server(
        LinearClassificationService(), server
    )
    server.add_insecure_port('[::]:86')
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
