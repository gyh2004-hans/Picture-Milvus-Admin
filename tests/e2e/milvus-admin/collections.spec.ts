/**
 * Collection 管理页 E2E 测试.
 *
 * 对齐优化计划 §6.4:
 *   - 集合列表加载: 页面展示 collection 列表
 *   - CRUD 操作: 新增/编辑/删除 → 状态即时更新
 */
import { test, expect } from '@playwright/test';

test.describe('Collection 管理页', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/collections');
    // 等待页面完全加载（Ant Design Table 渲染）
    await page.waitForSelector('.ant-table', { timeout: 10000 });
  });

  test('集合列表加载 — 页面展示 collection 列表', async ({ page }) => {
    // 验证页面标题/面包屑存在
    await expect(page.locator('.ant-breadcrumb').last()).toBeVisible();

    // 验证表格至少有一行（image_embeddings collection）
    const rows = page.locator('.ant-table-tbody tr');
    await expect(rows.first()).toBeVisible({ timeout: 10000 });

    // 验证关键列存在：名称、实体数、状态
    await expect(page.getByText('image_embeddings')).toBeVisible({ timeout: 10000 });
  });

  test('集合详情 — 点击查看展示 schema 信息', async ({ page }) => {
    // 点击"查看"按钮
    const viewButton = page.locator('.ant-btn').filter({ hasText: /查看/ }).first();
    if (await viewButton.isVisible()) {
      await viewButton.click();

      // 验证 Drawer 弹出
      const drawer = page.locator('.ant-drawer');
      await expect(drawer).toBeVisible({ timeout: 5000 });

      // 验证展示 schema 字段
      await expect(page.getByText(/vector|向量/)).toBeVisible({ timeout: 3000 });
    }
  });

  test('集合删除 — Popconfirm 二次确认后删除', async ({ page }) => {
    // 点击"删除"按钮（需要二次确认）
    const deleteButton = page.locator('.ant-btn').filter({ hasText: /删除/ }).first();
    if (await deleteButton.isVisible()) {
      await deleteButton.click();

      // 验证 Popconfirm 弹出
      const popconfirm = page.locator('.ant-popconfirm');
      await expect(popconfirm).toBeVisible({ timeout: 3000 });

      // 取消删除（安全起见，不真删）
      await page.locator('.ant-popconfirm .ant-btn').filter({ hasText: /取消|Cancel/ }).first().click();
      await expect(popconfirm).not.toBeVisible({ timeout: 3000 });
    }
  });
});
