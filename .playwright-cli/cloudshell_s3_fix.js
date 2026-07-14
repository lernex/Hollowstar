async page => {
  const input = page.locator('.ace_text-input');
  const command = `BUCKET="lernex-metis-artifacts-151025633969-us-east-1"; printf '%s' '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}' >/tmp/metis14-sse.json; aws s3api put-bucket-encryption --bucket "$BUCKET" --server-side-encryption-configuration file:///tmp/metis14-sse.json >/dev/null; aws s3api get-bucket-encryption --bucket "$BUCKET" --query "ServerSideEncryptionConfiguration.Rules[0].ApplyServerSideEncryptionByDefault.SSEAlgorithm" --output text; aws s3api get-bucket-versioning --bucket "$BUCKET" --query "Status" --output text; echo "S3_ROOT=s3://$BUCKET/metis14"`;
  await input.focus();
  await page.keyboard.type(command);
  await page.keyboard.press('Enter');
  await page.waitForTimeout(5000);
}
