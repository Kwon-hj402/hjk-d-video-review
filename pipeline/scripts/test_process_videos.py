"""Path/config/failure CLI fixtures in disposable work directories only."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import process_videos as batch


class ProcessVideoTests(unittest.TestCase):
    def setUp(self):
        work = batch.ROOT/'work'
        work.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='batch CLI tests ',dir=work)
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(work.resolve()))
        self.source = self.root/'입력 영상 폴더'
        self.source.mkdir()
        self.project = self.root/'project'
        (self.project/'config').mkdir(parents=True)
        self.weights = self.project/'fresh model.pt'
        self.weights.write_bytes(b'fake checkpoint, never loaded')

    def tearDown(self):
        # Verify the generated target remains under work before recursive cleanup.
        self.assertTrue(self.root.is_relative_to((batch.ROOT/'work').resolve()))
        self.temporary.cleanup()

    def video(self,name='영상 one.AVI',folder=None):
        parent = folder or self.source
        parent.mkdir(parents=True,exist_ok=True)
        path = parent/name
        path.write_bytes(b'fake source, never decoded')
        return path

    def test_spaces_recursive_and_resolved_dedup(self):
        a = self.video()
        b = self.video('영상 two.Mp4',self.source/'하위 폴더')
        result = batch.discover_sources([self.source,a])
        self.assertEqual(set(result['sources']),{a.resolve(),b.resolve()})
        self.assertEqual(len(result['skipped_duplicate_sources']),1)
        args = batch.parser().parse_args(['--source',str(self.source),str(a),'--weights',str(self.weights)])
        self.assertEqual(args.source,[str(self.source),str(a)])

    def test_missing_or_unsupported_source_prevents_writes(self):
        self.video()
        with self.assertRaises(ValueError):
            batch.discover_sources([self.source,self.root/'not present'])
        text = self.source/'text.txt';text.write_text('fixture',encoding='utf-8')
        with self.assertRaises(ValueError):
            batch.discover_sources([text])

    def test_prior_and_current_outputs_are_pruned(self):
        source = self.video()
        prior = self.source/'옛 분석'
        self.video('옛_D_비교.mp4',prior)
        (prior/batch.OUTPUT_MARKER).write_text('{}',encoding='utf-8')
        current = self.source/'새 분석'
        self.video('새_D_비교.mp4',current)
        result = batch.discover_sources([self.source],current)
        self.assertEqual(result['sources'],[source.resolve()])
        self.assertEqual(len(result['skipped_output_directories']),2)
        with self.assertRaises(ValueError):
            batch.discover_sources([prior])

    def test_dynamic_weights_keys_and_missing_config(self):
        with self.assertRaises(ValueError):
            batch.resolve_weights(project_root=self.project)
        for key in ('weights','trained_weights','best_checkpoint'):
            config = {key:'fresh model.pt','approved_snapshot':'snapshot fixture','approved_positive_count':17}
            (self.project/'config/inference_latest.json').write_text(json.dumps(config),encoding='utf-8')
            resolved, info = batch.resolve_weights(project_root=self.project)
            self.assertEqual(resolved,self.weights.resolve())
            self.assertEqual(info['approved_snapshot_info']['approved_positive_count'],17)
        with self.assertRaises(ValueError):
            batch.resolve_weights(self.root/'missing.pt',self.project)
        resolved,_ = batch.resolve_weights(self.weights,self.project)
        self.assertEqual(resolved,self.weights.resolve())

    def test_same_stem_uses_stable_path_hash(self):
        a = self.video('동일 영상.AVI',self.source/'A')
        b = self.video('동일 영상.AVI',self.source/'B')
        sources = batch.discover_sources([self.source])['sources']
        names = batch.video_directory_names(sources)
        self.assertNotEqual(names[a],names[b])
        self.assertTrue(all('__' in name for name in names.values()))
        self.assertEqual(names,batch.video_directory_names(sources))

    @staticmethod
    def renderer(**kwargs):
        source, output = kwargs['source'],kwargs['output']
        if source.stem.startswith('bad'):
            raise RuntimeError('intentional fixture failure')
        output.mkdir(exist_ok=False)
        paths = {'comparison_video':output/f'{source.stem}_D_비교.mp4',
                 'diameter_video':output/f'{source.stem}_D_표시.mp4',
                 'csv':output/f'{source.stem}_D_값.csv'}
        for path in paths.values():
            path.write_bytes(b'fixture output')
        return dict(state='partial' if kwargs['max_frames'] else 'complete',source=str(source),
                    processed=2,total=20,measured=1,unmeasurable=1,
                    **{key:str(value) for key,value in paths.items()})

    def test_failure_is_recorded_and_other_video_continues(self):
        a,b = self.video('bad source.AVI'),self.video('good source.AVI')
        output = self.root/'분석 결과'
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            result = batch.run_batch([self.source],output,self.weights,renderer=self.renderer,project_root=self.project)
        self.assertEqual(result['state'],'complete_with_errors')
        self.assertEqual(result['failed_video_count'],1)
        self.assertEqual(result['completed_video_count'],1)
        self.assertEqual(result['processed_frame_count'],2)
        self.assertFalse(result['fmd_calculated'])
        self.assertTrue((output/batch.OUTPUT_MARKER).is_file())
        self.assertTrue((output/'batch_report.json').is_file())
        html = (output/'index.html').read_text(encoding='utf-8')
        self.assertIn('good source.AVI',html)
        self.assertIn('처리 실패',html)
        self.assertIn('good%20source_D_',html)
        self.assertEqual(a.read_bytes(),b'fake source, never decoded')
        self.assertEqual(b.read_bytes(),b'fake source, never decoded')
        with self.assertRaises(ValueError):
            batch.run_batch([self.source],output,self.weights,renderer=self.renderer,project_root=self.project)

    def test_partial_is_allowed_only_for_explicit_smoke_limit(self):
        source = self.video('good.AVI')
        def partial_renderer(**kwargs):
            result = self.renderer(**kwargs)
            result['state']='partial'
            return result
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            full = batch.run_batch([source],self.root/'full',self.weights,renderer=partial_renderer,project_root=self.project)
            smoke = batch.run_batch([source],self.root/'smoke',self.weights,max_frames=2,renderer=partial_renderer,project_root=self.project)
        self.assertEqual(full['failed_video_count'],1)
        self.assertEqual(smoke['failed_video_count'],0)
        self.assertEqual(smoke['partial_video_count'],1)
        self.assertEqual(smoke['state'],'partial')
        self.assertTrue(smoke['smoke_test_only'])

    def test_cli_exit_codes_and_missing_weights_no_output(self):
        source = self.video()
        output = self.root/'must not be created'
        with mock.patch.object(batch,'resolve_weights',side_effect=ValueError('weights missing')):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(batch.main(['--source',str(source),'--output',str(output)]),2)
        self.assertFalse(output.exists())
        with mock.patch.object(batch,'run_batch',return_value={'failed_video_count':1}) as runner:
            self.assertEqual(batch.main(['--source',str(source),'--roi','94,40,545,382']),1)
            self.assertEqual(runner.call_args.args[5],[94.0,40.0,545.0,382.0])
        with mock.patch.object(batch,'run_batch',return_value={'failed_video_count':0}):
            self.assertEqual(batch.main(['--source',str(source),'--max-frames','2']),0)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ProcessVideoTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(tests_run=result.testsRun,successful=result.wasSuccessful(),failures=len(result.failures),errors=len(result.errors),
                  patient_data_or_models_loaded=False,fixture_location='work temporary directories',
                  covered=['spaces','recursive discovery','dedup','nonexisting inputs','output markers','dynamic model keys',
                           'basename collisions','failure continuation','no source overwrite','partial smoke gating','CLI exit codes'])
    (batch.ROOT/'reports/process_videos_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
