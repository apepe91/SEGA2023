import SimpleITK
import numpy as np
import trimesh
from pathlib import Path

from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

class Segaalgorithm(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            input_path=Path('/input/images/ct/'),
            output_path=Path('/output/'),
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
        self._segmentation_output_path = self._output_path / "images" / "aorta-segmentation"
        if not self._segmentation_output_path.exists():
            self._segmentation_output_path.mkdir(parents=True)



    def process_case(self, *, idx, case):
        # Load and test the image for this case
        input_image, input_image_file_path = self._load_input_image(case=case)

        # Segment (and mesh)
        predictions = self.predict(input_image=input_image)
        aorta_segmentation = predictions[0]
        visualization_mesh = predictions[1]     # surface to be used for visualization
        volume_mesh = predictions[2]            # surface to be converted to volume mesh

        # Write resulting segmentation to output locations

        segmentation_path = self._segmentation_output_path / input_image_file_path.name
        visualization_path = self._output_path / "aortic-vessel-tree.obj"
        mesh_path = self._output_path / "aortic-vessel-tree-volume-mesh.obj"

        SimpleITK.WriteImage(aorta_segmentation, str(segmentation_path), True)
        trimesh.exchange.export.export_mesh(visualization_mesh, str(visualization_path), 'obj')
        trimesh.exchange.export.export_mesh(volume_mesh, str(mesh_path), 'obj')

        # Write segmentation file path to 'result.json' for this case
        return {
            "outputs": [
                dict(type="metaio_image", filename=segmentation_path.name),
                dict(type="wavefront", filename=visualization_path.name),
                dict(type="wavefront", filename=mesh_path.name)
            ],
            "inputs": [
                dict(type="metaio_image", filename=input_image_file_path.name)
            ],
            "error_messages": [],
        }

    def predict(self, *, input_image: SimpleITK.Image) -> SimpleITK.Image:
        # Example: Segment all values greater than 2 in the input image
        outputs = [SimpleITK.BinaryThreshold(image1=input_image, lowerThreshold=2, insideValue=1, outsideValue=0),
                   trimesh.primitives.Box(),  #optional visualization task, leave Box() as placeholder if you do not participate
                   trimesh.primitives.Box()]  #optional volumetric meshing task, leave Box() as placeholder if you do not participate
        return outputs


if __name__ == "__main__":
    Segaalgorithm().process()
