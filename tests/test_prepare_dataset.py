from PIL import ImageFilter

from ml.training.prepare_dataset import hamming, parse_fragment_type, perceptual_hash
from tests.helpers import texture

ALIASES = {
    "epidermis bawah dengan idiobalas berupa sel minyak": "epidermis bawah dengan idioblas berupa sel minyak"
}


def test_parse_fragment_type_strips_prefix_indices_and_case():
    assert parse_fragment_type("cropped_Rambut Penutup 28(1).png", {}) == "rambut penutup"
    assert (
        parse_fragment_type("cropped_Idioblas berupa sel minyak 11(2).png", {})
        == "idioblas berupa sel minyak"
    )
    assert parse_fragment_type("cropped_BERKAS PENGANGKUT.png", {}) == "berkas pengangkut"


def test_parse_fragment_type_applies_spelling_aliases():
    name = "cropped_Epidermis bawah dengan Idiobalas berupa sel minyak 10.png"
    assert parse_fragment_type(name, ALIASES) == "epidermis bawah dengan idioblas berupa sel minyak"


def test_perceptual_hash_is_stable_under_mild_blur_and_differs_across_textures():
    image = texture("beta", 0)
    assert hamming(perceptual_hash(image), perceptual_hash(image)) == 0
    assert hamming(perceptual_hash(image), perceptual_hash(image.filter(ImageFilter.GaussianBlur(1)))) <= 6
    assert hamming(perceptual_hash(image), perceptual_hash(texture("alpha", 0))) > 16


def test_hamming_counts_differing_bits():
    assert hamming(0b1011, 0b0010) == 2
