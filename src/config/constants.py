"""
Project Constants Definition

This module centralizes all global constants used throughout the project.
It includes:

- Dataset class labels
- Professional multi-language medical prompts for CLIP-based models

"""

# ==================== Dataset Class Configuration ====================

# The order of this list determines the class index mapping.
CLASS_NAMES = [
    "Glioma",                    
    "Meningioma",               
    "NORMAL",                    
    "Neurocitoma",              
    "Outros Tipos de Lesões",   
    "Schwannoma" 
]

# Total number of classification categories
NUM_CLASSES = len(CLASS_NAMES)


# ==================== Professional Medical Prompts ====================
# Multi-language radiological descriptions used for CLIP text encoding.

# Languages included:
# - English (radiology-style reporting language)
# - Chinese
# - Portuguese

PROFESSIONAL_MEDICAL_PROMPTS = {
    "Glioma": [
        "MRI showing intra-axial glioma with infiltrative margins",
        "FLAIR hyperintense glioma in cerebral white matter", 
        "Ring-enhancing mass suspicious for high-grade glioma",
        "脑MRI提示胶质瘤，边界浸润，T2高信号",
        "Lesão intra-axial compatível com glioma na RM"
    ],

    "Meningioma": [
        "Extra-axial mass consistent with meningioma on MRI",
        "Dural-based lesion with homogeneous enhancement",
        "Convexity meningioma with dural tail sign",
        "颅内脑膜瘤，硬膜附着，均匀强化",
        "Meningioma extra-axial com realce homogêneo"
    ],

    "NORMAL": [
        "Normal brain MRI without pathological findings",
        "No evidence of mass lesion or abnormal enhancement",
        "Brain MRI shows normal anatomy and signal intensity",
        "正常脑MRI，未见异常信号或占位",
        "RM cerebral normal sem alterações patológicas"
    ],

    "Neurocitoma": [
        "Intraventricular mass consistent with neurocytoma",
        "Central neurocytoma in lateral ventricle",
        "侧脑室内神经细胞瘤，边界清楚",
        "Neurocitoma intraventricular com bordas bem definidas"
    ],

    "Outros Tipos de Lesões": [
        "Miscellaneous intracranial lesions on MRI",
        "Other types of cerebral pathological findings",
        "其他类型的颅内病变，MRI表现多样",
        "Outras lesões intracranianas de diferentes etiologias"
    ],

    "Schwannoma": [
        "Extra-axial schwannoma with cystic components",
        "Vestibular schwannoma in cerebellopontine angle",
        "小脑桥脑角神经鞘瘤，边界清楚",
        "Schwannoma do ângulo pontocerebelar com realce intenso"
    ],
}


# ==================== Utility Functions ====================

def get_class_index(class_name):
    try:
        return CLASS_NAMES.index(class_name)
    except ValueError:
        return -1


def get_class_name(index):

    if 0 <= index < NUM_CLASSES:
        return CLASS_NAMES[index]
    return None


def get_prompts_for_class(class_name):
    
    return PROFESSIONAL_MEDICAL_PROMPTS.get(class_name, [])


# ==================== BraTS Disease-Activity Prompts ====================
# Multi-language radiological descriptions for the 3-class active-disease
# scheme used in the BraTS-GLI 2024 follow-on experiment.
#
# Classes (must match scripts/inventory_brats.py CLASS_NAMES):
#   0 - Quiescent (minimal active disease)
#   1 - Enhancing without necrosis
#   2 - Necrotic enhancing (ring-enhancing)

BRATS_CLASS_NAMES = [
    "Quiescent (minimal active disease)",
    "Enhancing without necrosis",
    "Necrotic enhancing (ring-enhancing)",
]
BRATS_NUM_CLASSES = len(BRATS_CLASS_NAMES)

BRATS_DISEASE_ACTIVITY_PROMPTS = {
    "Quiescent (minimal active disease)": [
        "Post-operative brain MRI with stable resection cavity and no residual enhancement",
        "Follow-up MRI demonstrates surgical cavity without nodular enhancing tissue",
        "Treatment-effect changes on MRI without evidence of active glioma recurrence",
        "术后MRI显示切除腔，无明显强化复发",
        "RM pos-operatoria com cavidade de ressecao sem realce residual",
    ],

    "Enhancing without necrosis": [
        "Post-operative MRI with enhancing recurrent tumor adjacent to resection cavity",
        "Nodular enhancing recurrence without significant central necrosis on MRI",
        "Brain MRI shows active enhancing tumor along the surgical cavity wall",
        "术后MRI显示结节状强化复发肿瘤，无明显坏死",
        "Recidiva tumoral com realce nodular sem necrose evidente na RM",
    ],

    "Necrotic enhancing (ring-enhancing)": [
        "Ring-enhancing recurrent glioblastoma with central necrosis on post-operative MRI",
        "Necrotic enhancing mass consistent with high-grade glioma recurrence",
        "Aggressive necrotic ring-enhancing lesion adjacent to the resection cavity",
        "环形强化复发病灶伴中央坏死，提示高级别胶质瘤复发",
        "Lesao com realce em anel e necrose central sugestiva de recidiva de alto grau",
    ],
}


def get_brats_prompts_for_class(class_name):
    return BRATS_DISEASE_ACTIVITY_PROMPTS.get(class_name, [])
