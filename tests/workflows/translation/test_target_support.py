"""Regression checks for shared request identity and metadata infrastructure."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / ".github/scripts/translation"))

from request_identity import capture_request_mapping, internal_id
from capture_metadata import capture_config

class TargetSupportTests(unittest.TestCase):
    def test_exact_identity_and_restoration_on_error(self):
        processor = SimpleNamespace(add_request=lambda request: None)
        original = processor.add_request
        engine = SimpleNamespace(output_processor=processor)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                with capture_request_mapping(engine, Path(tmp) / 'identity.json') as mapping:
                    processor.add_request(SimpleNamespace(external_req_id='user-with-hyphens',request_id='unrelated-opaque-id'))
                    self.assertEqual(internal_id(mapping,'user-with-hyphens'),'unrelated-opaque-id')
                    with self.assertRaises(RuntimeError): internal_id(mapping,'missing')
                    raise ValueError('generation failure')
            self.assertIs(processor.add_request, original)
            self.assertTrue((Path(tmp)/'identity.json').is_file())
        with self.assertRaises(RuntimeError): internal_id({'0':['a','b']},'0')

    def test_metadata_schema_and_nonstandard_scaling(self):
        dimensions={'head_dim':128,'num_attention_heads':16,'num_key_value_heads':2,'attention_multiplier':128**-0.5}
        config=capture_config(SimpleNamespace(hidden_size=2048),dimensions)
        self.assertEqual(set(config),{'head_dim','hidden_size','num_attention_heads','num_key_value_heads','attention_multiplier'})
        dimensions['attention_multiplier']=0.5
        with self.assertRaises(RuntimeError): capture_config(SimpleNamespace(hidden_size=2048),dimensions)

if __name__ == '__main__': unittest.main()
