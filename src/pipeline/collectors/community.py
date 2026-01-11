import uuid
import datetime
import os
import logging
from urllib.parse import quote
from playwright.async_api import async_playwright
from dotenv import load_dotenv

load_dotenv()

# 로깅 설정
logger = logging.getLogger(__name__)

class InstizCollector:
    def __init__(self):
        self.source = "instiz"
        self.base_url = "https://www.instiz.net"
        self.user_id = os.getenv("INSTIZ_ID")
        self.user_pw = os.getenv("INSTIZ_PW")
    # 로그인 자동화 함수 
    async def login(self, page):
        try:
            # 홈페이지로 이동한 후 로딩 대기 
            await page.goto(self.base_url, wait_until="domcontentloaded", timeout=60000)
            await page.wait_for_timeout(3000)
            
            # 로그인 창 보이게 하고 아이디 입력칸에 포커스 
            await page.evaluate("""
                document.getElementById('loginwindow').style.display = 'block';
                document.getElementById('user_id').focus();
            """)
            await page.wait_for_timeout(1500)
            
            await page.evaluate(f"""
                document.getElementById('user_id').value = '{self.user_id}';
                document.getElementById('password').value = '{self.user_pw}';
            """)
            
            await page.wait_for_timeout(500)
            
            await page.evaluate("""
                document.querySelector('input[type="submit"].login_go').click();
            """)
            
            await page.wait_for_timeout(6000)
            
            page_text = await page.content()
            return "autologinok=1" in page_text
            
        except Exception as e:
            logger.error(f"로그인 중 오류 발생: {e}")
            return False
    # 게시글 수집 함수 
    async def collect(self, keyword: str, max_results: int = 5):
        results = []
        browser = None
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    # 실제 브라우저처럼 지정해서 차단 회피 
                    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                )
                page = await context.new_page()

                # 로그인 후 검색 페이지 이동 
                await self.login(page)
                encoded_keyword = quote(keyword)
                search_url = f"{self.base_url}/name_enter?category=2&k={encoded_keyword}&stype=9"
                await page.goto(search_url, wait_until="domcontentloaded", timeout=60000)
                await page.wait_for_timeout(4000)
                
                # 게시글 목록 테이블에서 제목 링크들  선택 
                post_links = await page.locator("tr[id^='list'] td.listsubject a").all()

                detail_urls = []
                for link in post_links[:max_results]:
                    try:
                        href = await link.get_attribute("href")
                        if href and "name_enter" in href:
                            if not href.startswith("http"):
                                href = self.base_url + href
                            detail_urls.append(href)
                    except Exception as e:
                        logger.warning(f"링크 추출 중 오류: {e}")
                # 게시글 상세 수집 (각 링크마다 새 페이지 열기)
                for url in detail_urls:
                    detail_page = None
                    try:
                        detail_page = await context.new_page()
                        await detail_page.goto(url, wait_until="domcontentloaded", timeout=60000)
                        await detail_page.wait_for_timeout(2000)

                        raw_title = await detail_page.evaluate("""
                            () => {
                                const elem = document.getElementById('nowsubject');
                                return elem ? elem.textContent.split('\\n')[0] : '';
                            }
                        """)

                        raw_content = await detail_page.evaluate("""
                            () => {
                                const elem = document.getElementById('memo_content_1');
                                return elem ? elem.textContent : '';
                            }
                        """)

                        raw_views = await detail_page.evaluate("""
                            () => {
                                const elem = document.getElementById('hit');
                                return elem ? parseInt(elem.textContent) || 0 : 0;
                            }
                        """)

                        raw_likes = await detail_page.evaluate("""
                            () => {
                                const elem = document.querySelector('.votenow98473432');
                                return elem ? parseInt(elem.textContent) || 0 : 0;
                            }
                        """)

                        standardized_data = self._map_to_standard_schema(
                            raw_title, raw_content, raw_views, raw_likes, "unknown", url, keyword
                        )
                        
                        results.append(standardized_data)

                    except Exception as e:
                        logger.error(f"게시글 수집 중 오류 (URL: {url}): {e}")
                    finally:
                        if detail_page:
                            await detail_page.close()

                await context.close()

        except Exception as e:
            logger.error(f"수집 중 오류 발생: {e}")
        finally:
            if browser:
                await browser.close()

        return results
    # 표준 스키마로 변환 
    def _map_to_standard_schema(self, title: str, content: str, views: int, likes: int, author_id: str, url: str, keyword: str) -> dict:
        return {
            "message_id": str(uuid.uuid4()),
            "type": "post",
            "source": "community",
            "collected_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
            "keyword": keyword,
            "content_data": {
                "text": f"{title}\n\n{content.strip()}"[:2000] if content else title,
                "author_id": author_id,
                "metrics": {
                    "views": views,
                    "likes": likes
                }
            }
        }