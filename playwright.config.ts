import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright E2E 测试配置 —— Milvus Admin 前端联调测试.
 *
 * 对齐优化计划 §6.4:
 *   前端 → apps/milvus-admin (Vite dev, port 5173)
 *   后端 → uvicorn src.main:app --reload (port 8000)
 *
 * 运行:
 *   npx playwright test tests/e2e/milvus-admin/
 *   npx playwright test --ui                    # 交互模式
 *   npx playwright test --project=chromium      # 仅 Chrome
 */
export default defineConfig({
  testDir: './tests/e2e/milvus-admin',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: [
    ['html', { outputFolder: 'playwright-report' }],
    ['list'],
  ],
  timeout: 30000,
  expect: {
    timeout: 10000,
  },

  use: {
    baseURL: process.env.E2E_BASE_URL || 'http://localhost:5173',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },

  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
  ],

  // 前端 dev server（自动启动/关闭）
  webServer: {
    command: 'cd apps/milvus-admin && npm run dev',
    url: 'http://localhost:5173',
    reuseExistingServer: !process.env.CI,
    timeout: 30000,
  },
});
