"""Exercise the UI against an isolated QA database, never real judgments."""
import json,shutil,threading
from pathlib import Path
from http.server import ThreadingHTTPServer
from playwright.sync_api import sync_playwright
from experiments.dog_domain.visual_study_server import make_handler,connect

root=Path('tmp/visual_study_qa');root.mkdir(parents=True,exist_ok=True)
source=Path('D:/meongtamjeong_research/human_visual_study_v1')
for name in ['study.json','access.json']:shutil.copy2(source/name,root/name)
shutil.copytree(source/'assets',root/'assets',dirs_exist_ok=True)
connect(root).close();access=json.loads((root/'access.json').read_text())
server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(root));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
url=f'http://127.0.0.1:{server.server_port}/#'
try:
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1100});errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(url+access['raters']['R1']);page.locator('#start').click()
        first_index=page.evaluate('index')
        first_pair=page.evaluate('state.tasks[index].id')
        page.locator('[data-grade="2"]').click()
        page.wait_for_function('(i)=>index===i+1 && !busy',arg=first_index)
        page.locator('#prev').click()
        assert page.locator('[data-grade="2"]').get_attribute('aria-pressed')=='true'
        # A failed save must preserve the current pair and prior stored score.
        page.route('**/api/rating',lambda route:route.fulfill(status=503,content_type='application/json',body=json.dumps({'error':'Test save failure'})))
        page.locator('[data-grade="3"]').click()
        page.locator('#saveStatus').filter(has_text='Test save failure').wait_for()
        assert page.evaluate('state.tasks[index].id')==first_pair
        assert page.evaluate('ratings[state.tasks[index].id].score')==2
        page.unroute('**/api/rating')
        page.locator('[data-grade="3"]').click()
        page.wait_for_function('(i)=>index===i+1 && !busy',arg=first_index)
        page.locator('#prev').click()
        assert page.locator('[data-grade="3"]').get_attribute('aria-pressed')=='true'
        # The final pair stays in bounds; an incomplete study is not called complete.
        page.evaluate('index=state.tasks.length-1;render()')
        page.locator('[data-grade="1"]').click()
        page.wait_for_function('!busy && ratings[state.tasks[index].id]?.score===1')
        assert page.evaluate('index===state.tasks.length-1')
        page.reload();page.locator('#reference').wait_for()
        assert page.locator('#identity').inner_text()=='평가자 R1'
        page.locator('#reference').click();assert page.locator('#zoom').is_visible();page.locator('#zoom button').click()
        page.screenshot(path=str(root/'rater_desktop.png'),full_page=True)
        with page.expect_download() as event:page.locator('#backup').click()
        event.value.save_as(str(root/'qa_export.json'))
        assert len(json.loads((root/'qa_export.json').read_text())['ratings'])>=1
        page.set_viewport_size({'width':390,'height':844});page.screenshot(path=str(root/'rater_mobile.png'),full_page=True)
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        admin=browser.new_page(viewport={'width':1440,'height':1100});admin.goto(url+access['admin']);admin.locator('#refresh').wait_for();admin.screenshot(path=str(root/'admin.png'),full_page=True)
        assert '비교 점수를 공개하지 않습니다' in admin.locator('main').inner_text()
        assert not errors,errors
        browser.close()
    print('PASS: isolated browser save, persistence, navigation, zoom, export, mobile layout, and blinded admin progress.')
finally:server.shutdown();server.server_close();thread.join()
