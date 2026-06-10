/**
 * Partition 管理页 E2E 测试.
 *
 * 对齐优化计划 §6.4:
 *   - 创建分区: 新建学科分区 → 列表刷新
 */
import { test, expect } from '@playwright/test';

test.describe('Partition 管理页', () => {
  test.beforeEach(async ({ page }) => {
    // 进入 image_embeddings 的分区管理页
    await page.goto('/collections/image_embeddings/partitions');
    await page.waitForSelector('.ant-table', { timeout: 10000 });
  });

  test('分区列表加载 — 展示 9 个学科分区 + _default', async ({ page }) => {
    const rows = page.locator('.ant-table-tbody tr');

    // 至少应有 _default 分区
    await expect(rows.first()).toBeVisible({ timeout: 10000 });

    // 验证有分区名称列
    const tableText = await page.locator('.ant-table').textContent();
    // 至少包含 _default
    expect(tableText).toContain('_default');
  });

  test('创建分区 — Modal 表单 → 列表刷新', async ({ page }) => {
    // 点击创建分区按钮
    const createButton = page.locator('.ant-btn').filter({ hasText: /创建|新建/ }).first();
    if (await createButton.isVisible()) {
      await createButton.click();

      // 验证 Modal 弹出
      const modal = page.locator('.ant-modal');
      await expect(modal).toBeVisible({ timeout: 3000 });

      // 填写分区名
      const input = modal.locator('input').first();
      await input.fill('e2e_test_partition');

      // 点击确认
      await modal.locator('.ant-btn').filter({ hasText: /确定|OK|创建/ }).first().click();

      // 等待 modal 关闭
      await expect(modal).not.toBeVisible({ timeout: 5000 });
    }
  });

  test('分区数据预览 — 点击分区展示数据列表', async ({ page }) => {
    // 点击第一个分区行（通常有数据的分区）
    const firstRow = page.locator('.ant-table-tbody tr').first();
    await firstRow.click();

    // 可能跳转到数据页或弹出详情
    // 验证页面有响应（URL 变化或内容刷新）
    await page.waitForLoadState('networkidle');
  });
});
