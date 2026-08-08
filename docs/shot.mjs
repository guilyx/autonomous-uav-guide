import { chromium } from 'playwright';
const B='http://localhost:4173/autonomous-uav-guide';
const out='/tmp/claude-0/-home-user-autonomous-uav-guide/5dd49567-9f47-5acd-bd1d-bac8b2fa687b/scratchpad';
const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium' });
for (const [name, url, dark] of [
  ['home-dark', '/', true],
  ['home-light', '/', false],
  ['fw-dark', '/vehicles/fixed-wing', true],
  ['learn-dark', '/learning/', true],
]) {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 },
    colorScheme: dark ? 'dark' : 'light' });
  await page.goto(B + url, { waitUntil: 'networkidle' });
  await page.waitForTimeout(1400);
  await page.screenshot({ path: `${out}/${name}.png`, fullPage: false });
  console.log('shot', name);
  await page.close();
}
await browser.close();
