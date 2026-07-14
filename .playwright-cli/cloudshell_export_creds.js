async page => {
  const input = page.getByRole('textbox', { name: 'CloudShell terminal input' });
  await input.focus();
  await page.keyboard.press('Control+A');
  await page.keyboard.type('aws configure export-credentials --format env-no-export || env | grep "^AWS_" || aws sts get-caller-identity');
  await page.keyboard.press('Enter');
  await page.waitForTimeout(5000);
}
