async page => {
  const input = page.locator('.ace_text-input');
  const command = 'ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text); REGION=us-east-1; BUCKET="lernex-metis-artifacts-${ACCOUNT_ID}-${REGION}"; if aws s3api head-bucket --bucket "$BUCKET" >/dev/null 2>&1; then echo "Bucket already exists: $BUCKET"; else aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" >/dev/null; echo "Created bucket: $BUCKET"; fi; aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true >/dev/null; aws s3api put-bucket-encryption --bucket "$BUCKET" --server-side-encryption-configuration "{\"Rules\":[{\"ApplyServerSideEncryptionByDefault\":{\"SSEAlgorithm\":\"AES256\"}}]}" >/dev/null; aws s3api put-bucket-versioning --bucket "$BUCKET" --versioning-configuration Status=Enabled >/dev/null; for key in metis14/ metis14/normalized-shards/ metis14/normalized-shards/pretrain/ metis14/normalized-shards/continued/ metis14/tokenizer/ metis14/pretrain-shards/ metis14/pretrain-shards/base/ metis14/pretrain-shards/continued/ metis14/chat-sft/ metis14/reasoning-sft/ metis14/reward-prefs/ metis14/dpo-prefs/ metis14/manifests/ metis14/checkpoints/ metis14/releases/; do aws s3api put-object --bucket "$BUCKET" --key "$key" >/dev/null; done; echo "S3_ROOT=s3://$BUCKET/metis14"';
  await input.focus();
  await page.keyboard.type(command);
  await page.keyboard.press('Enter');
  await page.waitForTimeout(9000);
}
