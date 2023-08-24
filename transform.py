# QONNX wrapper of ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper
# QONNX graph transformations for renaming and cleaning up
from qonnx.transformation.general import (
    GiveUniqueNodeNames,
    GiveReadableTensorNames,
    RemoveUnusedTensors,
    RemoveStaticGraphInputs
)
# QONNX graph transformations for inferring datatypes and shapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
# Precompute constant output nodes
from qonnx.transformation.fold_constants import FoldConstants
# Streamlining transformation: This is a collection of various transformations
from finn.transformation.streamline import Streamline
# Convert from QONNX model to FINN operators
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN


# Script entrypoint
if __name__ == '__main__':
    # Load the model graph
    model = ModelWrapper("attention.onnx")

    # Shape and datatype inference transformations
    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())

    # Some renaming and cleanup transformations
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveStaticGraphInputs())
    model = model.transform(RemoveUnusedTensors())

    # Fold constant-output nodes
    model = model.transform(FoldConstants())

    # # Try to apply streamlining transformation
    # #   Note: Does not work for attentions, as it assumes weight initializers
    # #   for matmul nodes
    # model = model.transform(Streamline())

    # Convert from QONNX graph to FINN nodes/operators
    model = model.transform(ConvertQONNXtoFINN())

    # Save the transformed graph
    model.save("attention.transformed.onnx")
