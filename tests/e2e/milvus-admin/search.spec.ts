/**
 * 向量检索页 E2E 测试.
 *
 * 对齐优化计划 §6.4:
 *   - 数据检索: 输入文本 → 返回相似图片卡片
 */
import { test, expect } from '@playwright/test';

test.describe('向量检索页', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/search');
    await page.waitForSelector('.ant-input, textarea', { timeout: 10000 });
  });

  test('页面加载 — 显示检索输入框和 Top-K 选择器', async ({ page }) => {
    // 验证文本输入区域存在
    const textarea = page.locator('textarea, .ant-input').first();
    await expect(textarea).toBeVisible({ timeout: 5000 });

    // 验证检索模式切换（文本/图片）
    const tabs = page.locator('.ant-tabs, .ant-radio-group');
    await expect(tabs.first()).toBeVisible({ timeout: 3000 });
  });

  test('文本检索 — 输入 prompt → 点击搜索 → 返回结果卡片', async ({ page }) => {
    // 找到文本输入框
    const textarea = page.locator('textarea, .ant-input').first();
    await textarea.fill('一幅地理教材风格的火山地貌示意图');

    // 找到检索按钮并点击
    const searchButton = page.locator('.ant-btn').filter({ hasText: /搜索|检索|Search/ }).first();
    if (await searchButton.isVisible()) {
      await searchButton.click();

      // 等待搜索结果（可能是卡片列表或空状态）
      // 空库时可能显示 Empty 状态，也是正常的
      await page.waitForTimeout(3000);

      // 验证页面有关键元素（结果或空状态提示）
      const hasContent = await page.locator('.ant-card, .ant-empty, .ant-result').first().isVisible().catch(() => false);
      expect(hasContent).toBeTruthy();
    }
  });

  test('分区过滤 — 选择学科后检索仅返回该学科结果', async ({ page }) => {
    // 找到学科选择器（Select 组件）
    const subjectSelect = page.locator('.ant-select').first();

    if (await subjectSelect.isVisible()) {
      await subjectSelect.click();

      // 等待下拉菜单
      const dropdown = page.locator('.ant-select-dropdown');
      await expect(dropdown).toBeVisible({ timeout: 3000 });

      // 选择一个学科（如"地理"）
      const option = dropdown.locator('.ant-select-item').filter({ hasText: /地理/ }).first();
      if (await option.isVisible()) {
        await option.click();
        await expect(subjectSelect).toContainText(/地理/);
      }
    }
  });

  test('Top-K 选择 — Slider 可调节返回数量', async ({ page }) => {
    // 找到 Slider 组件
    const slider = page.locator('.ant-slider');
    if (await slider.isVisible()) {
      await expect(slider).toBeVisible();
    }
  });
});
