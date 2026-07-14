async page => {
  const panel = page.getByRole('tabpanel');
  const text = await panel.textContent();
  return { text };
}
