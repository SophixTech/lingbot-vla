import importlib.util
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

spec = importlib.util.spec_from_file_location('all_frame_mask', Path(__file__).resolve().parents[1]/'mask/build_mask_all_frames.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class AllFrameMaskTests(unittest.TestCase):
    def test_box_coordinates(self):
        text = '[{"label":"robot_arm","bbox_format":"xyxy_1000","bbox_2d":[100,200,900,800]}]'
        self.assertEqual(m.parse_boxes(text, (768,480)), [[76.8,96.,691.2,384.]])
        self.assertEqual(m.parse_boxes('```json\n'+text+'\n```', (768,480)), m.parse_boxes(text,(768,480)))
        simple = text.replace('"bbox_format":"xyxy_1000",', '').replace('robot_arm', 'robotic arm')
        self.assertEqual(m.parse_boxes(simple, (768,480)), m.parse_boxes(text, (768,480)))
        with self.assertRaises(ValueError):
            m.parse_boxes(text.replace('xyxy_1000', 'pixel_xyxy'), (768,480))
        for bad in ['[]', text.replace('robot_arm','floor'), text.replace('900','1001'), text.replace('[100,200,900,800]','[900,200,100,800]')]:
            with self.assertRaises(ValueError):
                m.parse_boxes(bad, (768,480))

    def test_shards_cover_every_video_once(self):
        records = [('a',100),('b',20),('c',80),('d',15),('e',5)]
        jobs = m.assign_videos(records,2)
        self.assertCountEqual([r for job in jobs for r in job], records)
        self.assertEqual([sum(n for _, n in j) for j in jobs], [115,105])

    def test_decode_all_frames_and_export_only_two_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'dataset'; videos=root/'videos/cam'; videos.mkdir(parents=True)
            path=videos/'test.mp4'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10,(32,24))
            for i in range(7):
                writer.write(np.full((24,32,3),i*30,np.uint8))
            writer.release()
            records,size=m.index_videos([root],'cam')
            self.assertEqual(size,(32,24)); self.assertEqual(records[0][1],7)
            batches=list(m.frame_batches(str(path),7,3))
            self.assertEqual([i for i,_ in batches],[0,3,6])
            self.assertEqual(sum(len(b) for _,b in batches),7)
            with self.assertRaises(RuntimeError):
                list(m.frame_batches(str(path),8,3))
            out=Path(tmp)/'out';out.mkdir()
            keep=np.zeros((24,32),bool); keep[:,:16]=True
            m.save_result(out,keep,records,[root])
            self.assertEqual({p.name for p in out.iterdir()},{'mask.png','masked_preview.jpg'})
            np.testing.assert_array_equal(np.asarray(Image.open(out/'mask.png'))>0,keep)


if __name__ == '__main__':
    unittest.main()
