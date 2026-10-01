
from typing import List, Dict
from functools import lru_cache
from threading import Lock
import numpy as np
import trimesh
import torch
import smplx
import os

from measurement_definitions import *
from utils import *
from landmark_definitions import *
from joint_definitions import *


def set_shape(model, shape_coefs, device=None):
    '''
    Set shape of body model.
    :param model: smplx body model
    :param shape_coefs: torch.tensor dim (10,)
    :param device: device `model` lives on; defaults to the same
                   TORCH_DEVICE-derived default create_model() uses, since
                   this is only ever called right after building `model`
                   with that same default (see from_body_model below).
                   Not model.parameters()[0].device: whether smplx registers
                   betas/body_pose as nn.Parameter vs buffer isn't something
                   to guess at without a way to check against the installed
                   smplx version.

    Return
    shaped smplx body model
    '''
    shape_coefs = shape_coefs.to(device=device or _DEFAULT_DEVICE, dtype=torch.float32)
    with torch.no_grad():
        return model(betas=shape_coefs, return_verts=True)

def create_model(model_type, model_root, gender, num_betas=10, device=None):
    '''
    Create SMPL/SMPLX/etc. body model
    :param model_type: str of model type: smpl, smplx, etc.
    :param model_root: str of location where there are smpl/smplx/etc. folders with .pkl models
                        (clumsy definition in smplx package)
    :param gender: str of gender: MALE or FEMALE or NEUTRAL
    :param num_betas: int of number of shape coefficients
                      requires the model with num_coefs in model_root
    :param device: torch device string, e.g. "cpu" / "cuda" / "cuda:0".
                   Defaults to TORCH_DEVICE (env var) if set, else "cuda" if
                   available, else "cpu". fit_mesh.py's build_model() pins
                   this to "cpu" explicitly regardless of TORCH_DEVICE — see
                   the comment there for why.

    Return:
    :param smplx body model (SMPL, SMPLX, etc.), moved to `device`
    '''
    return _create_model_cached(model_type.lower(), model_root,
                                gender.upper(), num_betas,
                                device or _DEFAULT_DEVICE)


# TORCH_DEVICE lets you opt into GPU for the parts of this module that are
# already device-safe (this cache, set_shape, and therefore /measure/shape
# and MeasureBody.from_body_model). It does NOT affect fit_mesh.py's fit()
# loop, which stays CPU-only until that migration is done (see build_model()
# there). Auto-detecting "cuda" here is harmless either way: with no CUDA
# device present (true in this dev environment) torch.cuda.is_available()
# is False and this resolves to "cpu", identical to before this option existed.
_DEFAULT_DEVICE = os.environ.get("TORCH_DEVICE") or \
    ("cuda" if torch.cuda.is_available() else "cpu")

# Each SMPLX model retains a few hundred MB once loaded, so the cache is bounded
# rather than unbounded: 3 slots covers all genders of a single model type at
# one device. Raise SMPL_MODEL_CACHE if you serve smpl and smplx from one
# process, or mix devices (each (type, gender, device) combo is its own slot).
_MODEL_CACHE_SIZE = int(os.environ.get("SMPL_MODEL_CACHE", "3"))

# lru_cache alone doesn't stop a "cache stampede": two threads racing on the
# same not-yet-cached (model_type, gender, device) both see a miss and each
# load their own ~500MB copy of the pkl. This lock serializes model *builds*
# only (cheap no-op on every cache hit, since those never reach the lock's
# critical work more than once) — builds are rare (first request per gender)
# and a few seconds each, so one global lock is simpler than a per-key lock
# registry and the serialization cost is negligible.
_model_build_lock = Lock()


def _create_model_cached(model_type, model_root, gender, num_betas, device):
    '''Build (and memoize) a body model. Keyed on the args that actually change
    the result; smplx normalizes gender/model_type case internally, so the key is
    normalized too to avoid holding duplicate copies of the same model.
    '''
    with _model_build_lock:
        return _build_model(model_type, model_root, gender, num_betas, device)


@lru_cache(maxsize=_MODEL_CACHE_SIZE)
def _build_model(model_type, model_root, gender, num_betas, device):
    model = smplx.create(model_path=model_root,
                        model_type=model_type,
                        gender=gender,
                        use_face_contour=False,
                        num_betas=num_betas,
                        ext='pkl')
    return model.to(device)


@lru_cache(maxsize=None)
def _load_topology(model_type, body_model_path):
    '''Faces + face segmentation for a model type. These are identical for every
    gender and shape, so load them once per process instead of re-reading the
    ~500MB model pkl on every MeasureBody() construction.

    Reuses the NEUTRAL body model from the shared model cache (create_model)
    for its `.faces`, rather than loading a second, throwaway copy of the pkl
    just to read one array — faces are identical across gender for a given
    model_type, so any gender would do, but NEUTRAL doubles as a useful
    startup warm-up (see app.py's PRELOAD_GENDERS handling).
    '''
    model_root = os.path.dirname(body_model_path)
    faces = create_model(model_type, model_root, "NEUTRAL", num_betas=10).faces
    face_segmentation_path = os.path.join(body_model_path,
                                          f"{model_type}_body_parts_2_faces.json")
    return faces, load_face_segmentation(face_segmentation_path)


# Per-model-type constants for Measurer.__init__ (see the MeasureSMPL/MeasureSMPLX
# subclasses below). Built once at import time — not per MeasureBody() call — which
# also fixes SMPLMeasurementDefinitions()/SMPLXMeasurementDefinitions() previously
# being instantiated 4x on every single measurer construction.
def _model_config(landmarks, definitions_cls, joint2ind, num_joints, num_points):
    definitions = definitions_cls()
    return dict(
        landmarks=landmarks,
        length_definitions=definitions.LENGTHS,
        circumf_definitions=definitions.CIRCUMFERENCES,
        circumf_2_bodypart=definitions.CIRCUMFERENCE_TO_BODYPARTS,
        all_possible_measurements=definitions.possible_measurements,
        joint2ind=joint2ind,
        num_joints=num_joints,
        num_points=num_points,
    )


_MODEL_CONFIGS = {
    "smpl": _model_config(SMPL_LANDMARK_INDICES, SMPLMeasurementDefinitions,
                          SMPL_JOINT2IND, SMPL_NUM_JOINTS, 6890),
    "smplx": _model_config(SMPLX_LANDMARK_INDICES, SMPLXMeasurementDefinitions,
                           SMPLX_JOINT2IND, SMPLX_NUM_JOINTS, 10475),
}


class Measurer():
    '''
    Measure a parametric body model (SMPL, SMPLX, ...).

    All the measurements are expressed in cm. Concrete per-model-type behavior
    comes entirely from `_MODEL_CONFIGS[model_type]`; MeasureSMPL/MeasureSMPLX
    below are thin subclasses that just pick which config to use.
    '''

    def __init__(self, model_type: str):
        model_type = model_type.lower()
        cfg = _MODEL_CONFIGS[model_type]

        self.model_type = model_type
        self.body_model_root = "data"
        self.body_model_path = os.path.join(self.body_model_root, model_type)

        self.faces, self.face_segmentation = _load_topology(self.model_type,
                                                            self.body_model_path)

        self.landmarks = cfg["landmarks"]
        self.measurement_types = MEASUREMENT_TYPES
        self.length_definitions = cfg["length_definitions"]
        self.circumf_definitions = cfg["circumf_definitions"]
        self.circumf_2_bodypart = cfg["circumf_2_bodypart"]
        self.all_possible_measurements = cfg["all_possible_measurements"]
        self.joint2ind = cfg["joint2ind"]
        self.num_joints = cfg["num_joints"]
        self.num_points = cfg["num_points"]

        self.verts = None
        self.joints = None
        self.gender = None

        self.measurements = {}
        self.height_normalized_measurements = {}
        self.labeled_measurements = {}
        self.height_normalized_labeled_measurements = {}
        self.labels2names = {}

    def from_verts(self,
                   verts: torch.tensor):
        '''
        Construct body model from only vertices.
        :param verts: torch.tensor (self.num_points, 3) of model vertices
        '''

        verts = verts.squeeze()
        error_msg = f"verts need to be of dimension ({self.num_points},3)"
        assert verts.shape == torch.Size([self.num_points,3]), error_msg

        joint_regressor = get_joint_regressor(self.model_type,
                                              self.body_model_root,
                                              gender="NEUTRAL",
                                              num_thetas=self.num_joints)
        joints = torch.matmul(joint_regressor, verts)
        self.joints = joints.numpy()
        self.verts = verts.numpy()

    def from_body_model(self,
                        gender: str,
                        shape: torch.tensor):
        '''
        Construct body model from given gender and shape params.
        :param gender: str, MALE or FEMALE or NEUTRAL
        :param shape: torch.tensor, (1,10) beta parameters
        '''

        model = create_model(model_type=self.model_type,
                             model_root=self.body_model_root,
                             gender=gender,
                             num_betas=10)
        model_output = set_shape(model, shape)

        self.verts = model_output.vertices.detach().cpu().numpy().squeeze()
        self.joints = model_output.joints.squeeze().detach().cpu().numpy()
        self.gender = gender

    def measure(self,
                measurement_names: List[str]
                ):
        '''
        Measure the given measurement names from measurement_names list
        :param measurement_names - list of strings of defined measurements
                                    to measure from MeasurementDefinitions class
        '''

        for m_name in measurement_names:
            if m_name not in self.all_possible_measurements:
                print(f"Measurement {m_name} not defined.")
                continue

            if m_name in self.measurements:
                continue

            if self.measurement_types[m_name] == MeasurementType().LENGTH:

                value = self.measure_length(m_name)
                self.measurements[m_name] = value

            elif self.measurement_types[m_name] == MeasurementType().CIRCUMFERENCE:

                value = self.measure_circumference(m_name)
                self.measurements[m_name] = value

            else:
                print(f"Measurement {m_name} not defined")

    def measure_length(self, measurement_name: str):
        '''
        Measure distance between landmarks. If more than 2 landmarks are
        given, the landmarks are treated as a chain and the length is the
        sum of the distances between each consecutive pair (e.g. for
        landmarks A, B, C the length is dist(A,B) + dist(B,C)).
        :param measurement_name: str - defined in MeasurementDefinitions

        Returns
        :float of measurement in cm
        '''

        measurement_landmarks_inds = self.length_definitions[measurement_name]

        landmark_points = []
        for i in range(len(measurement_landmarks_inds)):
            if isinstance(measurement_landmarks_inds[i],tuple):
                # if touple of indices for landmark, take their average
                lm = (self.verts[measurement_landmarks_inds[i][0]] +
                          self.verts[measurement_landmarks_inds[i][1]]) / 2
            else:
                lm = self.verts[measurement_landmarks_inds[i]]

            landmark_points.append(lm)

        landmark_points = np.vstack(landmark_points)
        # chain consecutive points into segments: (N-1, 2, 3)
        segments = np.stack([landmark_points[:-1], landmark_points[1:]], axis=1)

        return self._get_dist(segments)

    @staticmethod
    def _get_dist(verts: np.ndarray) -> float:
        '''
        The Euclidean distance between vertices.
        The distance is found as the sum of each pair i
        of 3D vertices (i,0,:) and (i,1,:)
        :param verts: np.ndarray (N,2,3) - vertices used
                        to find distances

        Returns:
        :param dist: float, sumed distances between vertices
        '''

        verts_distances = np.linalg.norm(verts[:, 1] - verts[:, 0],axis=1)
        distance = np.sum(verts_distances)
        distance_cm = distance * 100 # convert to cm
        return distance_cm

    def measure_circumference(self,
                              measurement_name: str,
                              ):
        '''
        Measure circumferences. Circumferences are defined with
        landmarks and joints - the measurement is found by cutting the
        SMPL model with the  plane defined by a point (landmark point) and
        normal (vector connecting the two joints).
        :param measurement_name: str - measurement name

        Return
        float of measurement value in cm
        '''

        measurement_definition = self.circumf_definitions[measurement_name]
        circumf_landmarks = measurement_definition["LANDMARKS"]
        circumf_landmark_indices = [self.landmarks[l_name] for l_name in circumf_landmarks]
        circumf_n1, circumf_n2 = self.circumf_definitions[measurement_name]["JOINTS"]
        circumf_n1, circumf_n2 = self.joint2ind[circumf_n1], self.joint2ind[circumf_n2]

        plane_origin = np.mean(self.verts[circumf_landmark_indices,:],axis=0)
        plane_normal = self.joints[circumf_n1,:] - self.joints[circumf_n2,:]

        mesh = self._get_trimesh()

        # new version
        slice_segments, sliced_faces = trimesh.intersections.mesh_plane(mesh,
                                plane_normal=plane_normal,
                                plane_origin=plane_origin,
                                return_faces=True) # (N, 2, 3), (N,)

        slice_segments = filter_body_part_slices(slice_segments,
                                                 sliced_faces,
                                                 measurement_name,
                                                 self.circumf_2_bodypart,
                                                 self.face_segmentation)

        slice_segments_hull = convex_hull_from_3D_points(slice_segments)

        return self._get_dist(slice_segments_hull)

    def _get_trimesh(self):
        '''Trimesh of the current body, rebuilt only when self.verts changes.'''
        cached = getattr(self, "_trimesh_cache", None)
        if cached is None or cached[0] is not self.verts:
            cached = (self.verts, trimesh.Trimesh(vertices=self.verts, faces=self.faces))
            self._trimesh_cache = cached
        return cached[1]

    def height_normalize_measurements(self, new_height: float):
        '''
        Scale all measurements so that the height measurement gets
        the value of new_height:
        new_measurement = (old_measurement / old_height) * new_height
        NOTE the measurements and body model remain unchanged, a new
        dictionary height_normalized_measurements is created.

        Input:
        :param new_height: float, the newly defined height.

        Return:
        self.height_normalized_measurements: dict of
                {measurement:value} pairs with
                height measurement = new_height, and other measurements
                scaled accordingly
        '''
        if self.measurements != {}:
            old_height = self.measurements["height"]
            for m_name, m_value in self.measurements.items():
                norm_value = (m_value / old_height) * new_height
                self.height_normalized_measurements[m_name] = norm_value

            if self.labeled_measurements != {}:
                for m_name, m_value in self.labeled_measurements.items():
                    norm_value = (m_value / old_height) * new_height
                    self.height_normalized_labeled_measurements[m_name] = norm_value

    def label_measurements(self,set_measurement_labels: Dict[str, str]):
        '''
        Create labeled_measurements dictionary with "label: x cm" structure
        for each given measurement.
        NOTE: This overwrites any prior labeling!

        :param set_measurement_labels: dict of labels and measurement names
                                        (example. {"A": "head_circumference"})
        '''

        if self.labeled_measurements != {}:
            print("Overwriting old labels")

        self.labeled_measurements = {}
        self.labels2names = {}

        for set_label, set_name in set_measurement_labels.items():

            if set_name not in self.all_possible_measurements:
                print(f"Measurement {set_name} not defined.")
                continue

            if set_name not in self.measurements.keys():
                self.measure([set_name])

            self.labeled_measurements[set_label] = self.measurements[set_name]
            self.labels2names[set_label] = set_name

    def visualize(self,
                 measurement_names: List[str] = [],
                 landmark_names: List[str] = [],
                 title="Measurement visualization",
                 visualize_body: bool = True,
                 visualize_landmarks: bool = True,
                 visualize_joints: bool = True,
                 visualize_measurements: bool=True):

        if measurement_names == []:
            measurement_names = self.all_possible_measurements

        if landmark_names == []:
            landmark_names = list(self.landmarks.keys())

        # imported lazily: plotly is only needed for interactive visualization,
        # not for the measurement/API path
        from visualize import Visualizer

        vizz = Visualizer(verts=self.verts,
                        faces=self.faces,
                        joints=self.joints,
                        landmarks=self.landmarks,
                        measurements=self.measurements,
                        measurement_types=self.measurement_types,
                        length_definitions=self.length_definitions,
                        circumf_definitions=self.circumf_definitions,
                        joint2ind=self.joint2ind,
                        circumf_2_bodypart=self.circumf_2_bodypart,
                        face_segmentation=self.face_segmentation,
                        visualize_body=visualize_body,
                        visualize_landmarks=visualize_landmarks,
                        visualize_joints=visualize_joints,
                        visualize_measurements=visualize_measurements,
                        title=title
                        )

        vizz.visualize(measurement_names=measurement_names,
                       landmark_names=landmark_names,
                       title=title)


class MeasureSMPL(Measurer):
    '''Measure the SMPL model (6890 vertices), from shape params or verts.'''

    def __init__(self):
        super().__init__("smpl")


class MeasureSMPLX(Measurer):
    '''Measure the SMPLX model (10475 vertices), from shape params or verts.'''

    def __init__(self):
        super().__init__("smplx")


class MeasureBody():
    def __new__(cls, model_type):
        model_type = model_type.lower()
        if model_type == 'smpl':
            return MeasureSMPL()
        elif model_type == 'smplx':
            return MeasureSMPLX()
        else:
            raise NotImplementedError("Model type not defined")
